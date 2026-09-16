# HAT-DP

This repository is the implementation accompanying the paper **“A Hierarchical Adversarial Training Framework for Diffusion Purifier Adaptation.”**

HAT-DP adapts a pretrained diffusion purifier with hierarchical adversarial threats while keeping the target classifier frozen.

## Repository structure

- `CIFAR10/`: CIFAR-10 experiments and diffusion-purifier implementation.
- `CIFAR100/`: CIFAR-100 experiments and diffusion-purifier implementation.
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
Train HAT-DP on CIFAR-10 using the pretrained CIFAR-10 diffusion purifier and the fixed target classifier.
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
#### Eval on BPDA+EOT
Evaluate the trained HAT-DP model on CIFAR-10 under the fully adaptive BPDA+EOT attack.
```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_hatdp_cifar10_strong_bpda_eot.py \
  --mode bpda_eot \
  --diffusion-checkpoint checkpoints/strong_supervised_addt_latest2_last.pt \
  --data-root CIFAR10 \
  --T 280 \
  --scale 92000 \
  --purify-ensemble 4 \
  --bpda-eps 0.031372549 \
  --bpda-step-size 0.007843137 \
  --bpda-steps 50 \
  --bpda-eot-attack-reps 15 \
  --bpda-eot-defense-reps 150 \
  --bpda-subset-size 512 \
  --subset-seed 42 \
  --device cuda:0
```
### Run on CIFAR-100
#### Pre-Train
Pre-train a CIFAR-100 unconditional DDPM prior by fine-tuning the CIFAR-10 pretrained diffusion checkpoint for 100 epochs.
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u train_cifar100_diffusion_prior.py \
  --data-root . \
  --source-checkpoint cifar10_uncond_50M_500K.pt \
  --output-dir checkpoints/cifar100_prior_from_cifar10_100ep_bs256 \
  --epochs 100 \
  --batch-size 128 \
  --grad-accum-steps 2 \
  --learning-rate 2e-4 \
  --num-workers 4 \
  --seed 42 \
  --device cuda
```
#### Train
Train HAT-DP on CIFAR-100 using the dataset-matched CIFAR-100 diffusion prior and the fixed WRN-28-10 classifier.
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u train_hatdp_cifar100.py \
  --base-checkpoint checkpoints/cifar100_prior_from_cifar10_100ep_bs256/cifar100_uncond_from_cifar10_100ep_ema.pt \
  --classifier-checkpoint checkpoint_cifar100_wrn2810_1net_standard_bar1.ckpt \
  --data-root . \
  --out-checkpoint checkpoints/cifar100_hatdp_wrn28_10_seed42.pt \
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
  --seed 42 \
  --device cuda:0
```
#### Eval on BPDA+EOT
```bash
CUDA_VISIBLE_DEVICES=0 python -u eval_hatdp_cifar100.py \
  --mode bpda_eot \
  --diffusion-checkpoint checkpoints/cifar100_hatdp_wrn28_10_seed42.pt \
  --classifier-checkpoint checkpoint_cifar100_wrn2810_1net_standard_bar1.ckpt \
  --data-root . \
  --seeds 0 1 2 \
  --subset-size 512 \
  --T 280 \
  --scale 92000 \
  --purify-ensemble 4 \
  --bpda-eps 0.031372549 \
  --bpda-step-size 0.007843137 \
  --bpda-steps 50 \
  --bpda-eot-attack-reps 15 \
  --bpda-eot-defense-reps 150 \
  --device cuda:0
```
### Run on ImageNet
#### Train
```bash
cd /path/to/imagenet

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -u train_resnet50_dual_branch_100_per_class.py \
  --train-dir imagenet_train_100_per_class/train \
  --base-checkpoint 256x256_diffusion_uncond.pt \
  --out-checkpoint checkpoints/resnet50_v2_pgd_eot_4_255_100_bs2_100style_v1.pt \
  --resnet-weights v2 \
  --steps 20000 \
  --bs 2 \
  --lr 2e-6 \
  --train-output-blocks 5 \
  --t-min 150 \
  --t-max 600 \
  --delta-eps 0.015686275 \
  --adv-eps 0.015686275 \
  --adv-step-size 0.003921569 \
  --eot-eps 0.015686275 \
  --train-pgd-eot-steps 20 \
  --train-pgd-eot-iters 2 \
  --train-pgd-eot-step-size 0.003921569 \
  --num-workers 8 \
  --seed 42
```
#### Eval on PGD+EOT
```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --mode pgd+eot \
  --data-dir imagenet_val \
  --diffusion-checkpoint checkpoints/resnet50_v2_pgd_eot_4_255_100_bs2_100style_v1.pt \
  --model resnet50 \
  --resnet-weights v2 \
  --max-samples 512 \
  --T 110 \
  --scale 2000 \
  --bs 2 \
  --eps 0.015686275 \
  --pgd-steps 20 \
  --pgd-step-size 0.007843137 \
  --pgd-eot-iters 20 \
  --purify-ensemble 8 \
  --seed 42 \
  --output-dir res/hatdp_pgd_eot_seed42
```
## Citation

Citation information will be added after publication.
