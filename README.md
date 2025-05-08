# ADC-Net (EAAI-25-684) 
PyTorch implementation of **An Unsupervised Anatomy-aware Dual-constraint Cascade Network for Lung Computed Tomography Deformable Image Registration**

## Introduction
In this study, we propose an unsupervised Anatomy-aware Dual-constraint Cascade network called ADC-Net. An efficient single-level network based on an FCN is developed, incorporating a tri-directional deformation capture (TDC) block designed to calculate voxel correlations along three spatial dimensions separately, thereby comprehensively capturing lung deformations in different directions, particularly large deformations along the lung’s vertical axis. A low-computational-cost upsampling structure is adopted by the single-level network to ensure the efficiency and rapid performance of ADC-Net. To further improve the awareness of large deformations as well as internal anatomical details, we propose a dual-constraint cascade mode. In this mode, contour-enhanced and vessel-enhanced images used as constraints, along with the original images, are fed into three single-level networks, which achieves accurate registration of large deformations while ensuring precise alignment of internal details. ADC-Net was trained on three public lung CT datasets, SPARE, Creatis, and COPD, and tested on the Dir-lab and ThoraxCBCT datasets. Comparative experiments with several state-of-the-art methods thoroughly validate the performance of ADC-Net in the DIR of lung CT.
<div align="center">
  <img src="/fig2.png">
</div>

<div align="center">
  <img src="/fig3.svg">
</div>

## News
2025.05.08: Test dataset and pre-trained models on Dir-lab/ThoraxCBCT have been uploaded in the release page. The training data will be uploaded later.  
2025.05.08: Code have been released.

## Installation
Please use the following command for installation.
```
#It is recommended to create a new environment
conda create -n ADC-Net python==3.8
conda activate ADC-Net

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

#Install packages and other dependencies
pip install -r requirements.txt
```
Code has been tested with Ubuntu 20.04, Python 3.8, PyTorch 2.1, and CUDA 11.8.

## Pre-trained Weights
We provide pre-trained weights in the release page. Please download the latest weights and put them in weights-adam directory.
