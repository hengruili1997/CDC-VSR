# =============================================================================
# CDC-VSR: Cross-Domain Continuity Prior for Video Super-Resolution
# Model Components
# =============================================================================
# Architecture:
#   HaarWaveletTransform  →  低层小波基础算子
#   LWCU                  →  小波域特征分解 + 可学习子带细化
#   ASWD                  →  自适应稀疏高频增强 (仅处理高频子带 LH/HL/HH)
#   DifferentialGlobalMemory (DGM) →  差分全局记忆传播
#   WaveletEnhanceBlock   →  LWCU + ASWD 的组合块
#   SecondOrderDeformableAlignment →  二阶可变形对齐
#   BasicVSRPlusPlusWavelet        →  主干网络 (CDC-VSR)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


# =============================================================================
# Section 1: Wavelet Foundation
# =============================================================================

class HaarWaveletTransform(nn.Module):
    """
    2D Haar Discrete Wavelet Transform (固定滤波器，不参与学习).

    分解: x  →  (LL, LH, HL, HH)
    重建: (LL, LH, HL, HH)  →  x

    Note (对应审稿人R1/R2关于ASWD公式的问题):
        LL 为低频子带，LH/HL/HH 为三个高频子带。
        ASWD 仅对高频子带 (LH, HL, HH) 做稀疏增强，LL 直通。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # ---------- 分解滤波器 ----------
        ll = torch.tensor([[0.5,  0.5], [ 0.5,  0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5,  0.5]])
        hl = torch.tensor([[-0.5,  0.5], [-0.5,  0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5,  0.5]])

        # shape: (4*C, 1, 2, 2)  → 分组卷积用
        dec = torch.stack([ll, lh, hl, hh], dim=0)           # (4, 2, 2)
        dec = dec.unsqueeze(1).repeat(channels, 1, 1, 1)     # (4*C, 1, 2, 2)
        self.register_buffer('dec_filters', dec)

        # ---------- 重建滤波器 ----------
        rec_ll = torch.tensor([[0.5,  0.5], [ 0.5,  0.5]])
        rec_lh = torch.tensor([[-0.5,  0.5], [-0.5,  0.5]])
        rec_hl = torch.tensor([[-0.5, -0.5], [ 0.5,  0.5]])
        rec_hh = torch.tensor([[0.5, -0.5], [-0.5,  0.5]])

        rec = torch.stack([rec_ll, rec_lh, rec_hl, rec_hh], dim=0)  # (4, 2, 2)
        rec = rec.unsqueeze(0).repeat(channels, 1, 1, 1)             # (C, 4, 2, 2)
        self.register_buffer('rec_filters', rec)

    # ------------------------------------------------------------------
    def dwt(self, x: torch.Tensor):
        """
        前向 DWT.

        Args:
            x: (B, C, H, W)

        Returns:
            ll, lh, hl, hh: 各 (B, C, H/2, W/2)
            size_info: (pad_h, pad_w, orig_h, orig_w)  重建时去 padding 用
        """
        b, c, h, w = x.shape

        # 保证输入为偶数尺寸
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 构建分组卷积滤波器: (4*C, 1, 2, 2)
        filters = (self.dec_filters
                   .view(4, 1, 1, 2, 2)
                   .repeat(1, c, 1, 1, 1)
                   .view(4 * c, 1, 2, 2))

        # stride=2 完成下采样分解
        coeffs = F.conv2d(x, filters, stride=2, groups=c)          # (B, 4C, H/2, W/2)
        coeffs = coeffs.view(b, 4, c, coeffs.size(2), coeffs.size(3))

        ll = coeffs[:, 0]   # 低频
        lh = coeffs[:, 1]   # 水平高频
        hl = coeffs[:, 2]   # 垂直高频
        hh = coeffs[:, 3]   # 对角高频

        return ll, lh, hl, hh, (pad_h, pad_w, h, w)

    # ------------------------------------------------------------------
    def idwt(self,
             ll: torch.Tensor,
             lh: torch.Tensor,
             hl: torch.Tensor,
             hh: torch.Tensor,
             orig_size: tuple) -> torch.Tensor:
        """
        逆 DWT (完美重建).

        Args:
            ll, lh, hl, hh: 各 (B, C, H, W)
            orig_size: (pad_h, pad_w, orig_h, orig_w)

        Returns:
            x: (B, C, 2H, 2W)  (去掉 padding 后的原始大小)
        """
        pad_h, pad_w, orig_h, orig_w = orig_size
        b, c, h, w = ll.shape

        # 合并子带: (B, 4C, H, W)
        coeffs = torch.stack([ll, lh, hl, hh], dim=1).view(b, 4 * c, h, w)

        # 重建滤波器: (4C, 1, 2, 2)
        filters = (self.rec_filters
                   .view(c, 4, 1, 2, 2)
                   .permute(1, 0, 2, 3, 4)
                   .reshape(4 * c, 1, 2, 2))

        x = F.conv_transpose2d(coeffs, filters, stride=2, groups=c)

        # 去掉 pad
        if pad_h or pad_w:
            x = x[:, :, :orig_h, :orig_w]

        return x


# =============================================================================
# Section 2: LWCU — Learnable Wavelet Convolution Unit
# =============================================================================

class LWCU(nn.Module):
    """
    Learnable Wavelet Convolution Unit (Sec. III-A).

    对输入特征做小波分解后：
      - LL 子带: 独立轻量卷积细化
      - LH+HL+HH 高频子带: 分组深度卷积联合细化

    forward()    →  返回子带字典 {'ll', 'lh', 'hl', 'hh', 'size'}
    reconstruct() →  从子带字典重建空域特征
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.wavelet = HaarWaveletTransform(channels)

        # LL 子带细化
        self.ll_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 高频子带联合细化 (分组深度卷积, 保持子带独立性)
        self.high_refine = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, 3, 1, 1, groups=3),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> dict:
        ll, lh, hl, hh, size_info = self.wavelet.dwt(x)

        # 细化 LL
        ll = self.ll_refine(ll)

        # 细化高频 (LH, HL, HH 一起处理)
        high = torch.cat([lh, hl, hh], dim=1)
        high = self.high_refine(high)
        lh, hl, hh = torch.chunk(high, 3, dim=1)

        return {'ll': ll, 'lh': lh, 'hl': hl, 'hh': hh, 'size': size_info}

    def reconstruct(self, subbands: dict) -> torch.Tensor:
        return self.wavelet.idwt(
            subbands['ll'], subbands['lh'],
            subbands['hl'], subbands['hh'],
            subbands['size']
        )


# =============================================================================
# Section 3: ASWD — Adaptive Sparse Wavelet Domain Enhancement
# =============================================================================

class ASWD(nn.Module):
    """
    Adaptive Sparse Wavelet Domain Enhancement (Sec. III-B).

    ★ 重要设计说明 (对应审稿人 R1 Eq.(10) 和 R2 的一致性质疑):
        ASWD **仅**对高频子带 (LH, HL, HH) 进行自适应稀疏增强。
        LL 子带直接通过，不做任何处理。
        这与论文图示一致，Eq.(10) 应理解为仅作用于高频子带。

    每个高频子带独立处理:
        1. 全局自适应软阈值 (可微)  →  稀疏化噪声
        2. 空间门控注意力           →  突出重要位置
        3. 轻量残差增强              →  补偿细节
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # ---------- 共享的子带处理模块 ----------
        # 全局阈值预测
        self.threshold_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 4), 1, 1),
            nn.Sigmoid()
        )

        # 空间注意力门控
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, max(channels // 4, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 4), 1, 1),
            nn.Sigmoid()
        )

        # 残差增强卷积
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        # 各高频子带的可学习增益 (初始化为小值，保证训练稳定)
        self.lh_weight = nn.Parameter(torch.tensor(0.3))
        self.hl_weight = nn.Parameter(torch.tensor(0.3))
        self.hh_weight = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    @staticmethod
    def soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """可微软阈值函数 (proximal operator of L1)."""
        return torch.sign(x) * F.relu(torch.abs(x) - tau)

    # ------------------------------------------------------------------
    def _enhance_single_band(self,
                             band: torch.Tensor,
                             weight: nn.Parameter) -> torch.Tensor:
        """
        对单个高频子带做自适应稀疏增强.

        Args:
            band:   (B, C, H, W)  某高频子带
            weight: 标量可学习参数

        Returns:
            增强后的子带，与输入同形状
        """
        # 1. 全局自适应阈值 (缩放到较小范围避免过度稀疏)
        tau = self.threshold_pred(band) * 0.1

        # 2. 软阈值稀疏化
        sparse = self.soft_threshold(band, tau)

        # 3. 空间门控
        gate = self.spatial_gate(band)

        # 4. 增强
        enhanced = self.enhance(gate * sparse)

        # 5. 残差融合 + 可学习增益
        return band + torch.sigmoid(weight) * enhanced

    # ------------------------------------------------------------------
    def forward(self, subbands: dict) -> dict:
        """
        Args:
            subbands: LWCU.forward() 的返回值字典

        Returns:
            enhanced: 同格式字典，LL 不变，高频子带经过增强
        """
        return {
            'll':  subbands['ll'],   # ← LL 直通，不做增强
            'lh':  self._enhance_single_band(subbands['lh'], self.lh_weight),
            'hl':  self._enhance_single_band(subbands['hl'], self.hl_weight),
            'hh':  self._enhance_single_band(subbands['hh'], self.hh_weight),
            'size': subbands['size']
        }


# =============================================================================
# Section 4: DGM — Differential Global Memory
# =============================================================================

class DifferentialGlobalMemory(nn.Module):
    """
    Differential Global Memory (Sec. III-C).

    在时序传播中维护全局记忆状态，通过差分编码捕捉帧间运动变化，
    结合 ConvGRU 风格的门控机制更新记忆，增强长程时序一致性。

    Args:
        channels: 特征通道数

    Inputs:
        feat:      当前帧特征   (B, C, H, W)
        memory:    前一步记忆   (B, C, H, W)  or None (首帧)
        feat_prev: 前一帧特征   (B, C, H, W)  or None (首帧)

    Returns:
        output:     增强后的当前特征  (B, C, H, W)
        new_memory: 更新后的记忆状态  (B, C, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # 差分特征编码 (concat: feat + diff → channels)
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        # ConvGRU 更新门
        self.update_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.Sigmoid()
        )

        # ConvGRU 重置门
        self.reset_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.Sigmoid()
        )

        # 候选记忆
        self.candidate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.Tanh()
        )

        # 输出融合
        self.output_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self,
                feat: torch.Tensor,
                memory: torch.Tensor | None,
                feat_prev: torch.Tensor | None = None):

        if memory is None:
            memory = torch.zeros_like(feat)

        # 差分信号
        diff = (feat - feat_prev) if feat_prev is not None else torch.zeros_like(feat)

        # 差分编码
        diff_feat = self.diff_encoder(torch.cat([feat, diff], dim=1))

        # ConvGRU 门控
        combined = torch.cat([feat, memory], dim=1)
        update   = self.update_gate(combined)
        reset    = self.reset_gate(combined)

        reset_memory = reset * memory
        candidate    = self.candidate(torch.cat([feat, reset_memory], dim=1))

        # 记忆更新 (融合差分信息，系数 0.1 保持稳定)
        new_memory = (1 - update) * memory + update * (candidate + 0.1 * diff_feat)

        # 输出融合 + 残差
        output = feat + self.output_fusion(torch.cat([feat, new_memory], dim=1))

        return output, new_memory


# =============================================================================
# Section 5: Composite Blocks
# =============================================================================

class WaveletEnhanceBlock(nn.Module):
    """
    小波域增强块 = LWCU + ASWD + IDWT + 可学习残差融合.

    gamma 初始化为 0，训练初期等价于恒等变换，保证训练稳定性。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.lwcu = LWCU(channels)
        self.aswd = ASWD(channels)
        # 可学习残差权重，初始为 0 → 等价于跳过该模块
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subbands = self.lwcu(x)
        enhanced = self.aswd(subbands)
        out = self.lwcu.reconstruct(enhanced)
        # 软残差: tanh(gamma) ∈ (-1, 1)，防止过拟合
        return x + torch.tanh(self.gamma) * (out - x)


class EnhancedResidualBlock(nn.Module):
    """
    增强残差块 = Conv + Channel Attention + (可选) WaveletEnhanceBlock.

    当 use_wavelet=True 时，在通道注意力后追加小波增强。
    """

    def __init__(self, channels: int, use_wavelet: bool = True):
        super().__init__()
        self.use_wavelet = use_wavelet

        self.conv1  = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2  = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu  = nn.LeakyReLU(0.1, inplace=True)

        # Channel Attention (SE-style)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 16, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 16, 4), channels, 1),
            nn.Sigmoid()
        )

        if use_wavelet:
            self.wavelet_enhance = WaveletEnhanceBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        out = out * self.ca(out)             # channel attention

        if self.use_wavelet:
            out = self.wavelet_enhance(out)  # wavelet enhance

        return out + x                       # 全局残差


class ResidualBlocksWithWavelet(nn.Module):
    """
    带小波增强的残差块堆叠.

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        num_blocks:   残差块数量
        wavelet_freq: 每隔 wavelet_freq 个块插入一次小波增强 (默认 3)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 wavelet_freq: int = 3):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.blocks = nn.ModuleList([
            EnhancedResidualBlock(
                out_channels,
                use_wavelet=(i % wavelet_freq == wavelet_freq - 1)
            )
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# Section 6: Second-Order Deformable Alignment
# =============================================================================

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """
    二阶可变形对齐模块 (来自 BasicVSR++, 保持原始实现).

    利用当前帧与前两帧的光流估计偏移，实现精准的可变形特征对齐。
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


# =============================================================================
# Section 7: Main Network — BasicVSRPlusPlusWavelet (CDC-VSR)
# =============================================================================

@BACKBONES.register_module()
class BasicVSRPlusPlusWavelet(nn.Module):
    """
    CDC-VSR: BasicVSR++ with Cross-Domain Continuity Prior.

    三大核心改进 (对应论文三个贡献):
        1. LWCU + ASWD: 小波域频谱细化 (仅高频子带增强)
        2. DGM:          差分全局记忆时序传播
        3. 跨域损失:     在 cdc_vsr_loss.py 中实现

    Args:
        mid_channels:           特征通道数 (default: 64)
        num_blocks:             每个传播分支的残差块数 (default: 7)
        max_residue_magnitude:  可变形对齐最大偏移 (default: 10)
        is_low_res_input:       输入是否为低分辨率 (default: True)
        spynet_pretrained:      SPyNet 预训练权重路径
        cpu_cache_length:       超过此帧数启用 CPU 缓存 (default: 100)
        wavelet_freq:           小波增强频率，每 n 个 block 用一次 (default: 3)
        use_dgm:                是否启用 DGM 模块 (default: True)
    """

    def __init__(self,
                 mid_channels: int = 64,
                 num_blocks: int = 7,
                 max_residue_magnitude: int = 10,
                 is_low_res_input: bool = True,
                 spynet_pretrained: str | None = None,
                 cpu_cache_length: int = 100,
                 wavelet_freq: int = 3,
                 use_dgm: bool = True):

        super().__init__()
        self.mid_channels     = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.use_dgm          = use_dgm

        # ── 光流网络 ──────────────────────────────────────────────────
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # ── 特征提取 ─────────────────────────────────────────────────
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithWavelet(
                3, mid_channels, 5, wavelet_freq)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                ResidualBlocksWithWavelet(mid_channels, mid_channels, 5, wavelet_freq)
            )

        # ── 四路传播分支 ──────────────────────────────────────────────
        self.deform_align = nn.ModuleDict()
        self.backbone     = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude
            )
            self.backbone[module] = ResidualBlocksWithWavelet(
                (2 + i) * mid_channels, mid_channels, num_blocks, wavelet_freq)

        # ── DGM (每路传播一个) ────────────────────────────────────────
        if use_dgm:
            self.dgm = nn.ModuleDict({
                m: DifferentialGlobalMemory(mid_channels)
                for m in modules
            })

        # ── 重建 + 上采样 ─────────────────────────────────────────────
        self.reconstruction = ResidualBlocksWithWavelet(
            5 * mid_channels, mid_channels, 5, wavelet_freq)

        # 上采样前的最终小波增强
        self.pre_upsample = WaveletEnhanceBlock(mid_channels)

        self.upsample1  = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2  = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr    = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last  = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu      = nn.LeakyReLU(0.1, inplace=True)

        self.is_mirror_extended = False

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────
    def check_if_mirror_extended(self, lqs: torch.Tensor):
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs: torch.Tensor):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1: ].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            if flows_forward is not None:
                flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    # ──────────────────────────────────────────────────────────────────
    # Propagation
    # ──────────────────────────────────────────────────────────────────
    def propagate(self, feats: dict, flows: torch.Tensor, module_name: str) -> dict:
        n, t, _, h, w = flows.size()

        frame_idx   = list(range(0, t + 1))
        flow_idx    = list(range(-1, t))
        mapping_idx = list(range(len(feats['spatial'])))
        mapping_idx = mapping_idx + mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx  = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)

        # DGM 状态初始化
        memory    = None
        feat_prev = None

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop    = feat_prop.cuda()

            # ── 二阶可变形对齐 ─────────────────────────────────────
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond      = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # ── Backbone 特征聚合 ─────────────────────────────────
            feat = ([feat_current] +
                    [feats[k][idx]
                     for k in feats if k not in ('spatial', module_name)] +
                    [feat_prop])
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat_cat  = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat_cat)

            # ── DGM 记忆更新 ──────────────────────────────────────
            if self.use_dgm:
                if self.cpu_cache:
                    memory    = memory.cuda()    if memory    is not None else None
                    feat_prev = feat_prev.cuda() if feat_prev is not None else None

                feat_prop, memory = self.dgm[module_name](feat_prop, memory, feat_prev)
                feat_prev = feat_prop.clone()

                if self.cpu_cache:
                    memory    = memory.cpu()
                    feat_prev = feat_prev.cpu()

            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    # ──────────────────────────────────────────────────────────────────
    # Upsample & Reconstruct
    # ──────────────────────────────────────────────────────────────────
    def upsample(self, lqs: torch.Tensor, feats: dict) -> torch.Tensor:
        """
        逐帧重建高分辨率输出.

        Note (对应审稿人 R1 关于 Eq.(24) 残差结构):
            最终输出 = HR分支 + 双线性上采样的LR，
            即全局残差连接，使网络学习残差细节而非完整图像。
        """
        outputs     = []
        num_outputs = len(feats['spatial'])
        mapping_idx = list(range(num_outputs)) + list(range(num_outputs))[::-1]

        for i in range(lqs.size(1)):
            # 聚合所有传播分支特征
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            # 重建 → 小波增强 → 上采样
            hr = self.reconstruction(hr)
            hr = self.pre_upsample(hr)   # 最终小波细化

            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            # ── 全局残差: hr_out = hr_delta + bicubic(lq) ────────
            # (Eq.(24) 对应的残差结构)
            if self.is_low_res_input:
                hr = hr + self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr = hr + lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────
    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lqs: (B, T, 3, H, W)  低分辨率视频序列

        Returns:
            (B, T, 3, 4H, 4W)  超分辨率视频序列
        """
        n, t, c, h, w = lqs.size()
        self.cpu_cache = (t > self.cpu_cache_length and lqs.is_cuda)

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic'
            ).view(n, t, c, h // 4, w // 4)

        self.check_if_mirror_extended(lqs)

        # ── 空间特征提取 ──────────────────────────────────────────
        feats = {}
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(t):
                feat = self.feat_extract(lqs[:, i]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h_, w_ = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h_, w_)
            feats['spatial'] = [feats_[:, i] for i in range(t)]

        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            f'LR input must be ≥ 64×64, got {lqs_downsample.size(3)}×{lqs_downsample.size(4)}')

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # ── 双向二阶传播 ──────────────────────────────────────────
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)

    # ──────────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────────
    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None, got {type(pretrained)}')
