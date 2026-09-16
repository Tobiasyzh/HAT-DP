#!/usr/bin/env python3
"""Fine-tune the CIFAR-100 diffusion prior with the full HAT-DP objective.

This is a dataset-specific adaptation of ``train_addt_best.py``.  The HAT-DP
algorithm and its CIFAR-10 hyperparameters are intentionally preserved; only
the dataset, base diffusion checkpoint, and frozen classifier are changed.

Full HAT-DP run:

    OMP_NUM_THREADS=4 python -u train_hatdp_cifar100.py

The final output is a raw improved-diffusion state dict, matching the format
consumed by the existing purification code.  The script refuses to use a
10-class classifier and performs a CIFAR-100 classifier-accuracy preflight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from mixmo_cifar100_wrn import load_mixmo_cifar100_classifier
from train_addt_best import (
    Args as DiffusionArgs,
    addt_loss,
    apply_last_layer_finetune,
    apply_lora_finetune,
    classification_loss,
    count_trainable_parameters,
    eot_purifier_attack,
    pgd_attack,
    run_cgpo,
    save_checkpoint,
    set_seed,
    supervised_adv_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full HAT-DP adaptation of the CIFAR-100 diffusion purifier."
    )
    parser.add_argument(
        "--base-checkpoint",
        default=(
            "checkpoints/cifar100_prior_from_cifar10_100ep_bs256/"
            "cifar100_uncond_from_cifar10_100ep_ema.pt"
        ),
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default=(
            "models/cifar100/MixMo/"
            "checkpoint_cifar100_wrn2810_1net_standard_bar1.ckpt"
        ),
    )
    parser.add_argument("--data-root", default=".")
    parser.add_argument(
        "--out-checkpoint",
        default="checkpoints/cifar100_hatdp_wrn28_10_seed42.pt",
    )
    parser.add_argument(
        "--finetune-mode", default="last", choices=["lora", "last", "full"]
    )
    parser.add_argument("--train-output-blocks", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=6)
    parser.add_argument("--lora-alpha", type=float, default=6.0)

    # Matched to the successful CIFAR-10 HAT-DP configuration.
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--t-min", type=int, default=150)
    parser.add_argument("--t-max", type=int, default=600)
    parser.add_argument("--lambda-unit", type=float, default=0.3)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--lambda-max", type=float, default=0.4)
    parser.add_argument("--addt-weight", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=0.3)
    parser.add_argument("--eot-weight", type=float, default=0.8)
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--cgpo-steps", type=int, default=2)
    parser.add_argument("--cgpo-lr", type=float, default=1 / 255)
    parser.add_argument("--delta-eps", type=float, default=8 / 255)
    parser.add_argument("--delta-init", type=float, default=1 / 255)
    parser.add_argument("--adv-steps", type=int, default=10)
    parser.add_argument("--adv-step-size", type=float, default=2 / 255)
    parser.add_argument("--adv-eps", type=float, default=8 / 255)
    parser.add_argument("--eot-steps", type=int, default=7)
    parser.add_argument("--eot-iters", type=int, default=3)
    parser.add_argument("--eot-step-size", type=float, default=2 / 255)
    parser.add_argument("--eot-eps", type=float, default=8 / 255)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--min-classifier-accuracy",
        type=float,
        default=0.78,
        help="Abort if clean CIFAR-100 top-1 is below this fraction.",
    )
    parser.add_argument(
        "--skip-classifier-eval",
        action="store_true",
        help="Skip the full 10k-image accuracy preflight (shape checks still run).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Save a raw intermediate purifier every N steps; 0 disables it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the requested final output and metadata.",
    )
    args = parser.parse_args()

    if args.steps <= 0 or args.bs <= 0:
        parser.error("--steps and --bs must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0 <= args.t_min <= args.t_max < DiffusionArgs.diffusion_steps:
        parser.error(
            f"Require 0 <= t-min <= t-max < {DiffusionArgs.diffusion_steps}"
        )
    if not 0 <= args.lambda_min <= args.lambda_max <= 1:
        parser.error("Require 0 <= lambda-min <= lambda-max <= 1")
    if args.lambda_unit < 0:
        parser.error("--lambda-unit must be non-negative")
    if args.cgpo_steps <= 0 or args.adv_steps <= 0:
        parser.error("CGPO and pixel-PGD step counts must be positive")
    if args.eot_steps <= 0 or args.eot_iters <= 0:
        parser.error("EOT step/iteration counts must be positive")
    for name in ("delta_eps", "adv_eps", "eot_eps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.min_classifier_accuracy <= 1:
        parser.error("--min-classifier-accuracy must be in [0,1]")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def save_purifier_atomic(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_checkpoint(model, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_classifier(
    classifier: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    classifier.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += classifier(images).argmax(dim=1).eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def apply_finetune_policy(args: argparse.Namespace, model: nn.Module) -> None:
    if args.finetune_mode == "lora":
        apply_lora_finetune(
            model,
            args.train_output_blocks,
            args.lora_rank,
            args.lora_alpha,
        )
    elif args.finetune_mode == "last":
        apply_last_layer_finetune(model, args.train_output_blocks)
    elif args.finetune_mode == "full":
        model.requires_grad_(True)
    else:
        raise ValueError(f"Unknown fine-tune mode: {args.finetune_mode}")


def make_data_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_set = torchvision.datasets.CIFAR100(
        args.data_root,
        train=True,
        download=False,
        transform=Compose([ToTensor()]),
    )
    test_set = torchvision.datasets.CIFAR100(
        args.data_root,
        train=False,
        download=False,
        transform=Compose([ToTensor()]),
    )
    if len(train_set) != 50000 or len(test_set) != 10000:
        raise RuntimeError(
            f"Unexpected CIFAR-100 sizes: train={len(train_set)}, test={len(test_set)}"
        )
    if len(train_set.classes) != 100:
        raise RuntimeError(f"Expected 100 classes, got {len(train_set.classes)}")

    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=max(128, args.bs),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader, len(train_set), len(test_set)


def intermediate_path(final_path: Path, step: int) -> Path:
    return final_path.with_name(f"{final_path.stem}_step_{step:06d}{final_path.suffix}")


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    classifier_path = Path(args.classifier_checkpoint).expanduser().resolve()
    output_path = Path(args.out_checkpoint).expanduser().resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metrics_path = output_path.with_suffix(output_path.suffix + ".csv")

    for label, path in (
        ("base diffusion checkpoint", base_path),
        ("CIFAR-100 classifier checkpoint", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not args.overwrite:
        collisions = [path for path in (output_path, metadata_path, metrics_path) if path.exists()]
        if collisions:
            raise FileExistsError(
                "Refusing to overwrite existing run artifacts: "
                + ", ".join(str(path) for path in collisions)
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        cuda_index = 0 if device.index is None else device.index
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.seed)

    train_loader, test_loader, train_size, test_size = make_data_loaders(args)
    classifier, classifier_info = load_mixmo_cifar100_classifier(
        str(classifier_path), device
    )
    classifier_accuracy = None
    if not args.skip_classifier_eval:
        classifier_accuracy = evaluate_classifier(classifier, test_loader, device)
        if classifier_accuracy < args.min_classifier_accuracy:
            raise RuntimeError(
                f"Classifier top-1 is only {100 * classifier_accuracy:.2f}%, below "
                f"the required {100 * args.min_classifier_accuracy:.2f}%."
            )
        print(
            f"Classifier preflight: CIFAR-100 clean top-1={100 * classifier_accuracy:.2f}%",
            flush=True,
        )
    classifier_info["verified_clean_test_top1"] = classifier_accuracy

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(DiffusionArgs(), model_and_diffusion_defaults().keys())
    )
    base_state = torch.load(base_path, map_location="cpu")
    if not isinstance(base_state, dict) or not base_state:
        raise RuntimeError("Base diffusion checkpoint is not a non-empty state dict")
    model.load_state_dict(base_state, strict=True)
    apply_finetune_policy(args, model)
    model.to(device).train()
    trainable_parameters, total_parameters = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )

    manifest: Dict[str, Any] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "experiment": "Full HAT-DP on CIFAR-100 with frozen WRN-28-10",
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "dataset": {
            "name": "CIFAR-100",
            "training_examples": train_size,
            "test_examples_for_classifier_preflight": test_size,
        },
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "classifier_checkpoint": str(classifier_path),
        "classifier_checkpoint_sha256": sha256_file(classifier_path),
        "classifier": classifier_info,
        "output_checkpoint": str(output_path),
        "diffusion_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "protocol_note": (
            "All HAT-DP optimization defaults are held equal to the CIFAR-10 "
            "train_addt_best.py run; only dataset-specific checkpoints and "
            "the official MixMo vanilla 100-class frozen target classifier differ."
        ),
    }
    write_json_atomic(manifest, metadata_path)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                "step",
                "loss",
                "addt_recon",
                "adv_recon",
                "eot_recon",
                "classification",
                "gradient_norm",
                "t_min",
                "t_max",
            ]
        )

    print("Full HAT-DP / CIFAR-100", flush=True)
    print(
        f"base={base_path} | classifier={classifier_path} | output={output_path}",
        flush=True,
    )
    print(
        f"steps={args.steps} batch={args.bs} lr={args.lr:g} t=[{args.t_min},{args.t_max}] "
        f"finetune={args.finetune_mode}",
        flush=True,
    )
    print(
        f"trainable={trainable_parameters:,}/{total_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.4f}%) | "
        f"weights(addt/adv/eot/semantic)="
        f"{args.addt_weight}/{args.adv_weight}/{args.eot_weight}/{args.cls_weight}",
        flush=True,
    )

    iterator = iter(train_loader)
    last_step = 0
    last_values: Dict[str, float] = {}
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True)
    try:
        for step in progress:
            last_step = step
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                images, labels = next(iterator)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x0 = Normalize(0.5, 0.5)(images)
            timesteps = torch.randint(
                args.t_min,
                args.t_max + 1,
                (x0.shape[0],),
                device=device,
            )

            pixel_adversarial = pgd_attack(
                classifier,
                images,
                labels,
                eps=args.adv_eps,
                step_size=args.adv_step_size,
                steps=args.adv_steps,
            )
            pixel_adversarial_normalized = Normalize(0.5, 0.5)(pixel_adversarial)

            eot_adversarial = eot_purifier_attack(
                args,
                diffusion,
                model,
                classifier,
                images,
                labels,
                device,
            )
            eot_adversarial_normalized = Normalize(0.5, 0.5)(eot_adversarial)

            delta = run_cgpo(
                args,
                diffusion,
                model,
                classifier,
                x0,
                labels,
                timesteps,
            )
            addt_recon, pred_addt = addt_loss(
                args, diffusion, model, x0, timesteps, delta
            )
            adv_recon, pred_adv = supervised_adv_loss(
                diffusion,
                model,
                x0,
                pixel_adversarial_normalized,
                timesteps,
            )
            eot_recon, pred_eot = supervised_adv_loss(
                diffusion,
                model,
                x0,
                eot_adversarial_normalized,
                timesteps,
            )

            semantic_loss = torch.zeros((), device=device)
            if args.cls_weight > 0:
                semantic_loss = (
                    classification_loss(classifier, pred_addt, labels)
                    + classification_loss(classifier, pred_adv, labels)
                    + classification_loss(classifier, pred_eot, labels)
                ) / 3.0

            loss = (
                args.addt_weight * addt_recon
                + args.adv_weight * adv_recon
                + args.eot_weight * eot_recon
                + args.cls_weight * semantic_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()

            last_values = {
                "loss": float(loss.detach()),
                "addt_recon": float(addt_recon.detach()),
                "adv_recon": float(adv_recon.detach()),
                "eot_recon": float(eot_recon.detach()),
                "classification": float(semantic_loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        step,
                        f"{last_values['loss']:.10g}",
                        f"{last_values['addt_recon']:.10g}",
                        f"{last_values['adv_recon']:.10g}",
                        f"{last_values['eot_recon']:.10g}",
                        f"{last_values['classification']:.10g}",
                        f"{last_values['gradient_norm']:.10g}",
                        int(timesteps.min()),
                        int(timesteps.max()),
                    ]
                )
            progress.set_postfix(
                loss=f"{last_values['loss']:.5f}",
                addt=f"{last_values['addt_recon']:.5f}",
                adv=f"{last_values['adv_recon']:.5f}",
                eot=f"{last_values['eot_recon']:.5f}",
                cls=f"{last_values['classification']:.4f}",
            )

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                snapshot_path = intermediate_path(output_path, step)
                if snapshot_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Intermediate checkpoint exists: {snapshot_path}")
                save_purifier_atomic(model, snapshot_path)
                manifest.update(
                    {
                        "last_completed_step": step,
                        "latest_intermediate_checkpoint": str(snapshot_path),
                        "latest_losses": last_values,
                    }
                )
                write_json_atomic(manifest, metadata_path)
                print(f"Saved intermediate checkpoint: {snapshot_path}", flush=True)

    except BaseException as exc:
        emergency_path = intermediate_path(output_path, last_step)
        if last_step > 0 and not emergency_path.exists():
            save_purifier_atomic(model, emergency_path)
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "stopped_at_utc": utc_now(),
                "last_completed_step": last_step,
                "latest_intermediate_checkpoint": (
                    str(emergency_path) if last_step > 0 else None
                ),
                "latest_losses": last_values,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json_atomic(manifest, metadata_path)
        raise

    save_purifier_atomic(model, output_path)
    peak_memory_gib = None
    if device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_steps": args.steps,
            "final_losses": last_values,
            "peak_cuda_memory_allocated_gib": peak_memory_gib,
            "output_checkpoint_sha256": sha256_file(output_path),
            "evaluation_status": "not_run",
        }
    )
    write_json_atomic(manifest, metadata_path)
    print(f"Saved final CIFAR-100 HAT-DP checkpoint: {output_path}", flush=True)
    print(f"SHA256: {manifest['output_checkpoint_sha256']}", flush=True)
    print("Purifier evaluation was intentionally not run here.", flush=True)


if __name__ == "__main__":
    main()
