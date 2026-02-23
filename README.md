# When to Trust a Diffusion Model: Conditioning Strength and Sampling Variability in High-Stakes Domains

Official PyTorch implementation of our paper (paper id: 1062) submitted to MICCAI 2026.

## Abstract

Diffusion models have achieved impressive success in image generation tasks, yet their inherent stochasticity raises concerns when applied to high-stakes domains such as healthcare, where generated images are expected to be consistent and reproducible given the conditioning input. This study investigates the relationship between task-specific conditioning strength and the sampling variability of conditional diffusion models. We evaluate three representative architectures: ControlNet-style DDPM, classifier-free guidance DDPM, and latent diffusion models. Focusing on common vision tasks in radiology imaging, we use quantitative metrics to assess both sampling variability and conditioning strength, revealing a consistent inverse relationship: stronger conditioning leads to more stable and less stochastic outputs. Our findings demonstrate that sampling variability is a general issue across diffusion model architectures rather than being model-specific. The results extend to text-to-image generation, confirming the generality of our findings. We conclude with actionable guidelines for reliable deployment of diffusion models in high-stakes settings, emphasizing the importance of variability quantification and conditioning strength assessment.

## Repository Structure

```
.
├── cfg/              # ControlNet-style DDPM implementation
├── ddpm/             # Classifier-free guidance DDPM implementation
└── ldm/              # Latent Diffusion Model implementation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 12.4+

## Training

The repository contains three different model implementations, each in its respective directory. Below are the commands to train each model:

### Training ControlNet-style DDPM

```bash
cd cfg
python train_ddpm.py --config_path ./configs/init-cfg-config.yaml
```

### Training Classifier-free Guidance DDPM

```bash
cd ddpm
python train_ddpm.py --config_path ./configs/init-ddpm-config.yaml
```

### Training Latent Diffusion Model

```bash
cd ldm
python train_ldm.py
```

## Configuration

Each model implementation has its own configuration file:
- ControlNet-style DDPM: `cfg/configs/init-cfg-config.yaml`
- Classifier-free Guidance DDPM: `ddpm/configs/init-ddpm-config.yaml`
- LDM implementation uses hardcoded parameters in the training script
