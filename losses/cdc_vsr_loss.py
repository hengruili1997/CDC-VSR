# =============================================================================
# CDC-VSR: Cross-Domain Continuity Prior for Video Super-Resolution
# Loss Functions
# =============================================================================
# 包含:
#   CharbonnierLoss    →  基础像素损失 (平滑 L1)
#   GradientLoss       →  Sobel 梯度域损失
#   WaveletLoss        →  小波域子带加权损失 (HH > HL ≈ LH >> LL)
#   TemporalLoss       →  光流引导时序一致性损失
#   PerceptualLoss     →  VGG 感知损失 (可选, 用于 GAN/感知模式)
#   CrossDomainVSRLoss →  统一跨域损失接口 (用于训练配置)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Section 1: Pixel-Domain Loss
# =============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss — 平滑 L1, 对离群点更鲁棒.

    L_char(x) = sqrt(x^2 + eps^2)

    Args:
        eps: 平滑项 (default: 1e-6)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


# =============================================================================
# Section 2: Gradient-Domain Loss
# =============================================================================

class GradientLoss(nn.Module):
    """
    Sobel 梯度域损失 — 增强边缘与纹理恢复.

    对 pred 和 target 分别计算 Sobel x/y 方向梯度图，
    用 Charbonnier Loss 衡量梯度差异。

    Args:
        eps: Charbonnier 平滑项
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self._char = CharbonnierLoss(eps)

    def _compute_grad(self, x: torch.Tensor):
        """计算多通道 Sobel 梯度."""
        c = x.size(1)
        kx = self.sobel_x.repeat(c, 1, 1, 1)
        ky = self.sobel_y.repeat(c, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx,   pred_gy   = self._compute_grad(pred)
        target_gx, target_gy = self._compute_grad(target)
        return self._char(pred_gx, target_gx) + self._char(pred_gy, target_gy)


# =============================================================================
# Section 3: Wavelet-Domain Loss
# =============================================================================

class WaveletLoss(nn.Module):
    """
    小波域子带加权损失 (Sec. IV).

    使用固定 Haar 小波将 pred/target 分解为 (LL, LH, HL, HH)，
    对各子带计算 Charbonnier Loss 并加权求和。

    子带权重设计原则 (与论文图一致):
        HH (对角高频) > HL ≈ LH (方向高频) >> LL (低频)
    LL 保留低权重而非置零，保证低频内容不被完全忽视。

    ★ 注意 (对应审稿人 R1/R2 关于 ASWD Eq.(10) 的讨论):
        损失函数中的小波变换独立于模型中的 ASWD 模块。
        损失作用于像素空间的 pred/target，
        ASWD 作用于中间特征空间，两者不冲突。

    Args:
        channels:       输入图像通道数 (default: 3 for RGB)
        subband_weights: 各子带损失权重字典
        eps:            Charbonnier 平滑项
    """

    # 默认子带权重
    DEFAULT_WEIGHTS = {'LL': 0.1, 'LH': 0.3, 'HL': 0.3, 'HH': 0.5}

    def __init__(self,
                 channels: int = 3,
                 subband_weights: dict | None = None,
                 eps: float = 1e-6):
        super().__init__()
        self.channels       = channels
        self.subband_weights = subband_weights or self.DEFAULT_WEIGHTS
        self._char          = CharbonnierLoss(eps)

        # ---------- 固定 Haar 滤波器 ----------
        ll = torch.tensor([[0.5,  0.5], [ 0.5,  0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5,  0.5]])
        hl = torch.tensor([[-0.5,  0.5], [-0.5,  0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5,  0.5]])

        dec = torch.stack([ll, lh, hl, hh], dim=0)           # (4, 2, 2)
        dec = dec.unsqueeze(1).repeat(channels, 1, 1, 1)     # (4C, 1, 2, 2)
        self.register_buffer('dec_filters', dec)

    def _dwt(self, x: torch.Tensor):
        """对输入 x: (B, C, H, W) 做 Haar DWT."""
        b, c, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        filters = (self.dec_filters
                   .view(4, 1, 1, 2, 2)
                   .repeat(1, c, 1, 1, 1)
                   .view(4 * c, 1, 2, 2))

        coeffs = F.conv2d(x, filters, stride=2, groups=c)
        coeffs = coeffs.view(b, 4, c, coeffs.size(2), coeffs.size(3))

        return {
            'LL': coeffs[:, 0],
            'LH': coeffs[:, 1],
            'HL': coeffs[:, 2],
            'HH': coeffs[:, 3],
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_bands   = self._dwt(pred)
        target_bands = self._dwt(target)

        total   = 0.0
        w_total = sum(self.subband_weights.values())

        for name, w in self.subband_weights.items():
            loss   = self._char(pred_bands[name], target_bands[name])
            total += w * loss

        return total / w_total


# =============================================================================
# Section 4: Temporal Consistency Loss
# =============================================================================

class TemporalLoss(nn.Module):
    """
    光流引导的时序一致性损失.

    对相邻帧用后向光流 warp，仅在非遮挡区域计算一致性约束。
    遮挡检测: 光流幅度超过阈值的区域视为遮挡，不参与损失计算。

    Args:
        occlusion_threshold: 光流幅度阈值 (像素, default: 50)
    """

    def __init__(self, occlusion_threshold: float = 50.0):
        super().__init__()
        self.occ_thresh = occlusion_threshold

    def forward(self,
                pred_seq: torch.Tensor,
                flow_backward: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_seq:      (B, T, C, H, W)
            flow_backward: (B, T-1, 2, H, W)

        Returns:
            时序一致性损失 (标量)
        """
        from mmedit.models.common import flow_warp  # lazy import 避免循环依赖

        b, t, c, h, w = pred_seq.size()
        if t <= 1:
            return pred_seq.new_zeros(1).squeeze()

        total, count = 0.0, 0
        for i in range(1, t):
            curr_frame = pred_seq[:, i]
            prev_frame = pred_seq[:, i - 1]
            flow       = flow_backward[:, i - 1]

            # Warp 前一帧到当前帧
            prev_warped = flow_warp(prev_frame, flow.permute(0, 2, 3, 1))

            # 遮挡掩码 (光流幅度大 → 遮挡 → 不计损失)
            flow_mag        = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2)
            occlusion_mask  = (flow_mag < self.occ_thresh).float()

            diff   = torch.abs(curr_frame - prev_warped) * occlusion_mask
            total += diff.mean()
            count += 1

        return total / max(count, 1)


# =============================================================================
# Section 5: Perceptual Loss (Optional)
# =============================================================================

class PerceptualLoss(nn.Module):
    """
    VGG-19 感知损失 (可选).

    默认关闭，在 PSNR 导向训练时无需使用。
    当 perceptual_weight > 0 且需要提升视觉质量时启用。

    提取层: relu1_2, relu2_2, relu3_4
    权重:   [0.1, 0.1, 1.0]  (深层权重更高)

    Args:
        layer_weights: 三层感知特征权重列表
    """

    def __init__(self, layer_weights: list | None = None):
        super().__init__()
        from torchvision import models

        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:18])  # relu3_4

        for param in self.parameters():
            param.requires_grad = False

        self.weights = layer_weights or [0.1, 0.1, 1.0]

        # ImageNet 归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_f1   = self.slice1(pred)
        pred_f2   = self.slice2(pred_f1)
        pred_f3   = self.slice3(pred_f2)

        with torch.no_grad():
            tgt_f1  = self.slice1(target)
            tgt_f2  = self.slice2(tgt_f1)
            tgt_f3  = self.slice3(tgt_f2)

        return (self.weights[0] * F.l1_loss(pred_f1, tgt_f1) +
                self.weights[1] * F.l1_loss(pred_f2, tgt_f2) +
                self.weights[2] * F.l1_loss(pred_f3, tgt_f3))


# =============================================================================
# Section 6: CrossDomainVSRLoss — Unified Training Loss
# =============================================================================

class CrossDomainVSRLoss(nn.Module):
    """
    跨域视频超分辨率统一损失函数 (用于训练配置文件).

    三域损失设计:
        L_total = w_pix * L_pixel
                + w_grad * L_gradient
                + w_wav  * L_wavelet
                [+ w_temp * L_temporal   (需提供 flow_backward)]
                [+ w_perc * L_perceptual (use_perceptual=True)]

    PCGrad 冲突感知梯度对齐: 需在优化器层面实现，
    可参考 https://github.com/WeiChengTseng/Pytorch-PCGrad

    Args:
        pixel_weight:       像素损失权重      (default: 1.0)
        gradient_weight:    梯度损失权重      (default: 0.1)
        wavelet_weight:     小波损失权重      (default: 0.1)
        temporal_weight:    时序一致性权重    (default: 0.0, 关闭)
        perceptual_weight:  感知损失权重      (default: 0.0, 关闭)
        use_perceptual:     是否加载 VGG     (default: False)
        eps:                Charbonnier eps  (default: 1e-6)

    Example (训练配置):
        loss = CrossDomainVSRLoss(
            pixel_weight=1.0,
            gradient_weight=0.1,
            wavelet_weight=0.1,
        )
        total_loss, loss_dict = loss(pred, target)
    """

    def __init__(self,
                 pixel_weight:      float = 1.0,
                 gradient_weight:   float = 0.1,
                 wavelet_weight:    float = 0.1,
                 temporal_weight:   float = 0.0,
                 perceptual_weight: float = 0.0,
                 use_perceptual:    bool  = False,
                 eps:               float = 1e-6):
        super().__init__()

        self.pixel_weight      = pixel_weight
        self.gradient_weight   = gradient_weight
        self.wavelet_weight    = wavelet_weight
        self.temporal_weight   = temporal_weight
        self.perceptual_weight = perceptual_weight

        # ── 子损失模块 ──────────────────────────────────────────────
        self.loss_pixel    = CharbonnierLoss(eps)
        self.loss_gradient = GradientLoss(eps)
        self.loss_wavelet  = WaveletLoss(channels=3, eps=eps)

        if temporal_weight > 0:
            self.loss_temporal = TemporalLoss()
        else:
            self.loss_temporal = None

        if use_perceptual and perceptual_weight > 0:
            self.loss_perceptual = PerceptualLoss()
        else:
            self.loss_perceptual = None

    # ------------------------------------------------------------------
    def _flatten_video(self, x: torch.Tensor):
        """将视频张量 (B,T,C,H,W) 展平为图像张量 (B*T,C,H,W)."""
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.view(b * t, c, h, w)
        return x

    # ------------------------------------------------------------------
    def forward(self,
                pred:          torch.Tensor,
                target:        torch.Tensor,
                flow_backward: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:          预测帧 (B, T, C, H, W) 或 (B, C, H, W)
            target:        Ground Truth，与 pred 同形状
            flow_backward: 后向光流 (B, T-1, 2, H, W)，可选，用于时序损失

        Returns:
            total_loss: 标量损失
            loss_dict:  各子损失的详细字典 (用于日志)
        """
        pred_seq    = pred    # 保留序列维度用于时序损失
        pred_flat   = self._flatten_video(pred)
        target_flat = self._flatten_video(target)

        # ── 空域损失 ─────────────────────────────────────────────────
        l_pixel    = self.loss_pixel(pred_flat, target_flat)
        l_gradient = self.loss_gradient(pred_flat, target_flat)
        l_wavelet  = self.loss_wavelet(pred_flat, target_flat)

        total_loss = (self.pixel_weight    * l_pixel    +
                      self.gradient_weight * l_gradient +
                      self.wavelet_weight  * l_wavelet)

        loss_dict = {
            'loss_pixel':    l_pixel.item(),
            'loss_gradient': l_gradient.item(),
            'loss_wavelet':  l_wavelet.item(),
        }

        # ── 时序损失 (可选) ──────────────────────────────────────────
        if (self.loss_temporal is not None and
                self.temporal_weight > 0 and
                flow_backward is not None and
                pred_seq.dim() == 5):
            l_temporal  = self.loss_temporal(pred_seq, flow_backward)
            total_loss += self.temporal_weight * l_temporal
            loss_dict['loss_temporal'] = l_temporal.item()
        else:
            loss_dict['loss_temporal'] = 0.0

        # ── 感知损失 (可选) ──────────────────────────────────────────
        if self.loss_perceptual is not None and self.perceptual_weight > 0:
            l_perceptual  = self.loss_perceptual(pred_flat, target_flat)
            total_loss   += self.perceptual_weight * l_perceptual
            loss_dict['loss_perceptual'] = l_perceptual.item()
        else:
            loss_dict['loss_perceptual'] = 0.0

        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict
