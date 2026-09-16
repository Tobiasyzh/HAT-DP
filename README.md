# HAT-DP

This repository is the implementation accompanying the paper **“A Hierarchical Adversarial Training Framework for Diffusion Purifier Adaptation.”**

HAT-DP adapts a pretrained diffusion purifier with hierarchical adversarial threats while keeping the target classifier frozen.

## Repository structure

- `CIFAR10/`: CIFAR-10 experiments and diffusion-purifier implementation.
- `ImageNet/`: ImageNet experiments and guided-diffusion implementation.
- `requirements.txt`: core Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```
## Pre-trained Models
You can download pretrained models here:

- DDPM on ImageNet [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)
  -  checkpoint [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)
- DDPM on Cifar10 [https://github.com/openai/improved-diffusion](https://github.com/openai/improved-diffusion)
  - checkpoint [cifar10_uncond_50M_500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)

## Run experiments
### Run on CIFAR-10
#### Train
```bash
cd /path/to/cifar10

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u train_addt_best_server_reference.py \
  --base-checkpoint cifar10_uncond_50M_500K.pt \
  --out-checkpoint checkpoints/strong_supervised_addt_latest2_last.pt \
  --finetune-mode last \
  --train-output-blocks 5 \
  --steps 5000 \
  --bs 16 \
  --lr 3e-6 \
  --t-min 150 \
  --t-max 600 \
  --lambda-unit 0.3 \
  --lambda-min 0.05 \
  --lambda-max 0.4 \
  --seed 42
```
#### Eval
### Run on CIFAR-100
#### Train
#### Eval
### Run on ImageNet
#### Train
#### Eval
## Citation

Citation information will be added after publication.
