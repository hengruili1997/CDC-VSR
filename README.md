# CDC-VSR: Unifying Cross-Domain Continuity for High-Quality Video Super-Resolution

## Project Overview

This project is designed for **Unifying Cross-Domain Continuity for High-Quality Video Super-Resolution**. It aims to improve the reconstruction quality of low-resolution videos by leveraging complementary information across different video domains, with a focus on recovering fine textures, preserving structural details, and maintaining temporal consistency.

The project is built upon the open-source image and video restoration framework **BasicSR**:

> BasicSR: https://github.com/XPixelGroup/BasicSR/tree/master

BasicSR provides a unified PyTorch-based framework for image and video restoration tasks, including super-resolution, denoising, deblurring, JPEG artifact removal, and video restoration. Based on this framework, the proposed project extends the model architecture, loss functions, and experiment configurations for cross-domain video super-resolution.

## Released Components

This repository provides the following components:

### 1. Model Code

The repository includes the implementation of the proposed cross-domain joint video super-resolution model. The model contains modules such as:

- feature extraction modules;
- temporal modeling modules;
- cross-domain feature fusion modules;
- reconstruction modules.

The model is integrated into the BasicSR framework through its registration mechanism, making it convenient for training, validation, testing, and model deployment within a unified pipeline.

### 2. Loss Functions

The repository provides the loss functions used during model training. These losses include:

- pixel-wise reconstruction loss;
- perceptual loss;
- temporal consistency loss;
- cross-domain constraint loss.

The loss functions can be flexibly configured in the training `.yml` files according to different experimental settings.

### 3. Training and Testing Configuration Files

The repository also includes `.yml` configuration files for both training and testing. These files define key experimental settings, including:

- dataset paths;
- degradation settings;
- model parameters;
- training iterations;
- optimizer settings;
- learning rate schedules;
- loss weights;
- validation settings;
- testing and result-saving paths.

Users can modify these configuration files to train, validate, and test the proposed video super-resolution model under different experimental conditions.

## Framework Dependency

This project depends on BasicSR. Please install and configure BasicSR before running the training or testing scripts.

```bash
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -r requirements.txt
python setup.py develop
