#!/usr/bin/env python3
"""Fine-tune the CIFAR-10 unconditional DDPM prior on CIFAR-100.

The ADDT paper and its official implementation fine-tune the CIFAR-10 DDPM on
CIFAR-100 for 100 epochs.  This script transfers that training recipe to the
OpenAI ``improved_diffusion`` checkpoint format used by this project, while
preserving the existing U-Net architecture and diffusion process so that the
result remains directly loadable by ``diffusion.DiffusionPurificationModel``.

Primary references:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/
      f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html
  https://github.com/LYMDLUT/ADDT/blob/main/train_DDPM_DDIM_ADDT/train_clean.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version
from torchvision import datasets, transforms

from diffusion import Args as DiffusionArgs
from improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


RECIPE_REFERENCE = {
    "paper": "Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective",
    "paper_url": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html",
    "official_code": "https://github.com/LYMDLUT/ADDT/tree/main/train_DDPM_DDIM_ADDT",
    "adaptation": (
        "The paper fine-tunes a CIFAR-10 DDPM on CIFAR-100 for 100 epochs. "
        "Its public clean-training script uses effective batch size 256, "
        "AdamW(lr=2e-4, betas=(0.95,0.999), weight_decay=1e-6), gradient "
        "clipping at 1.0, and warm-started EMA capped at 0.9999."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt the project's CIFAR-10 improved-diffusion prior to CIFAR-100."
    )
    parser.add_argument("--data-root", default=".")
    parser.add_argument(
        "--source-checkpoint", default="cifar10_uncond_50M_500K.pt"
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/cifar100_prior_from_cifar10"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.95)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-max-decay", type=float, default=0.9999)
    parser.add_argument("--ema-inv-gamma", type=float, default=1.0)
    parser.add_argument("--ema-power", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--state-every-epochs",
        type=int,
        default=5,
        help="Save a resumable latest_state.pt at this epoch interval.",
    )
    parser.add_argument(
        "--snapshot-every-epochs",
        type=int,
        default=10,
        help="Save an inference-compatible raw EMA state dict at this interval.",
    )
    parser.add_argument("--eval-every-epochs", type=int, default=10)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=10,
        help="Fixed CIFAR-100 test batches used only as a diagnostic, never for checkpoint selection.",
    )
    parser.add_argument("--log-every-updates", type=int, default=25)
    parser.add_argument(
        "--resume",
        default="",
        help="Path to a resumable state, or 'auto' for OUTPUT_DIR/latest_state.pt.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after this many optimizer updates; intended for a smoke test (0 means unlimited).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA fp16 autocast. The paper's reported clean fine-tuning used fp32.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-random-flip", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        parser.error("--batch-size and --grad-accum-steps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.ema_max_decay < 1:
        parser.error("--ema-max-decay must be in [0, 1)")
    if args.ema_inv_gamma <= 0 or args.ema_power <= 0:
        parser.error("EMA warm-up parameters must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_torch_save(payload: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cpu_state_dict(module: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, tensor.detach().cpu()) for name, tensor in module.state_dict().items()
    )


def nested_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(nested_to_cpu(item) for item in value)
    return value


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def unwrap_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """Accept a raw state dict and common wrapped checkpoint layouts."""
    if isinstance(payload, Mapping) and payload:
        if all(torch.is_tensor(value) for value in payload.values()):
            state = payload
        else:
            state = None
            for key in (
                "ema_model",
                "ema_state_dict",
                "model",
                "model_state_dict",
                "state_dict",
            ):
                candidate = payload.get(key)
                if isinstance(candidate, Mapping) and candidate and all(
                    torch.is_tensor(value) for value in candidate.values()
                ):
                    state = candidate
                    break
            if state is None:
                raise ValueError(
                    "Checkpoint is not a raw state dict and has no recognized model field."
                )
    else:
        raise ValueError("Checkpoint does not contain a non-empty mapping.")

    state = OrderedDict((str(key), value) for key, value in state.items())
    if state and all(key.startswith("module.") for key in state):
        state = OrderedDict((key[7:], value) for key, value in state.items())
    return state


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    args: argparse.Namespace, generator: torch.Generator, use_cuda: bool
) -> Tuple[DataLoader, DataLoader]:
    train_ops = []
    if not args.no_random_flip:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = datasets.CIFAR100(
        root=args.data_root,
        train=True,
        transform=transforms.Compose(train_ops),
        download=False,
    )
    eval_set = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        transform=eval_transform,
        download=False,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,
        # Re-create workers at each epoch so an epoch-boundary resume derives
        # the same worker seeds from the saved DataLoader generator state.
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    eval_loader = DataLoader(eval_set, shuffle=False, drop_last=False, **common)
    return train_loader, eval_loader


def ema_decay_at_step(
    optimization_step: int, max_decay: float, inv_gamma: float, power: float
) -> float:
    """Match Diffusers EMAModel(use_ema_warmup=True) used by official ADDT."""
    step = max(0, optimization_step - 1)
    if step <= 0:
        return 0.0
    value = 1.0 - (1.0 + step / inv_gamma) ** (-power)
    return min(value, max_decay)


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    optimization_step: int,
    max_decay: float,
    inv_gamma: float,
    power: float,
) -> float:
    decay = ema_decay_at_step(optimization_step, max_decay, inv_gamma, power)
    ema_params = list(ema_model.parameters())
    model_params = [parameter.detach() for parameter in model.parameters()]
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, model_params, alpha=1.0 - decay)
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(model_buffer)
    return decay


@torch.no_grad()
def diagnostic_loss(
    model: nn.Module,
    diffusion: Any,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    seed: int,
) -> Dict[str, float]:
    if max_batches <= 0:
        return {}
    was_training = model.training
    model.eval()
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    totals: MutableMapping[str, float] = {"loss": 0.0, "mse": 0.0, "vb": 0.0}
    sample_count = 0
    for batch_index, (images, _labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            images.shape,
            dtype=images.dtype,
            device=device,
            generator=generator,
        )
        terms = diffusion.training_losses(model, images, timesteps, noise=noise)
        for key in totals:
            if key in terms:
                totals[key] += terms[key].sum().item()
        sample_count += batch_size
    if was_training:
        model.train()
    if sample_count == 0:
        return {}
    return {key: value / sample_count for key, value in totals.items()}


def append_metrics(path: Path, row: Mapping[str, Any]) -> None:
    columns = [
        "utc_time",
        "epoch",
        "global_step",
        "train_loss",
        "train_mse",
        "train_vb",
        "ema_decay",
        "diagnostic_ema_loss",
        "diagnostic_ema_mse",
        "diagnostic_ema_vb",
        "epoch_seconds",
    ]
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def save_training_state(
    path: Path,
    epoch_completed: int,
    global_step: int,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    args: argparse.Namespace,
    source_sha256: str,
) -> None:
    state = {
        "format_version": 1,
        "saved_at_utc": utc_now(),
        "epoch_completed": epoch_completed,
        "global_step": global_step,
        "model": cpu_state_dict(model),
        "ema_model": cpu_state_dict(ema_model),
        "optimizer": nested_to_cpu(optimizer.state_dict()),
        "data_generator_state": data_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "args": vars(args),
        "source_checkpoint_sha256": source_sha256,
    }
    atomic_torch_save(state, path)


def restore_training_state(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    device: torch.device,
    expected_source_sha256: str,
) -> Tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    checkpoint_hash = state.get("source_checkpoint_sha256")
    if checkpoint_hash and checkpoint_hash != expected_source_sha256:
        raise RuntimeError(
            "Resume state was created from a different source checkpoint: "
            f"{checkpoint_hash} != {expected_source_sha256}"
        )
    model.load_state_dict(state["model"], strict=True)
    ema_model.load_state_dict(state["ema_model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_device(optimizer, device)
    data_generator.set_state(state["data_generator_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    return int(state["epoch_completed"]), int(state["global_step"])


def environment_manifest(
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    model: nn.Module,
    diffusion: Any,
) -> Dict[str, Any]:
    model_config = {
        key: getattr(DiffusionArgs(), key)
        for key in model_and_diffusion_defaults().keys()
    }
    return {
        "created_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd().resolve()),
        "arguments": vars(args),
        "recipe_reference": RECIPE_REFERENCE,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "model_config": model_config,
        "diffusion_timesteps": int(diffusion.num_timesteps),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision_version,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "notes": [
            "CIFAR-100 class labels are intentionally ignored: this is an unconditional prior.",
            "The test-set diffusion loss is diagnostic only and is never used for checkpoint selection.",
            "The final exported checkpoint is a raw EMA state dict compatible with diffusion.py.",
        ],
    }


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    if device.type == "cuda":
        cuda_index = 0 if device.index is None else device.index
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)

    seed_everything(args.seed, args.deterministic)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    print(f"[{utc_now()}] Hashing source checkpoint: {source_path}", flush=True)
    source_sha256 = sha256_file(source_path)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(DiffusionArgs(), model_and_diffusion_defaults().keys())
    )
    source_payload = torch.load(source_path, map_location="cpu")
    source_state = unwrap_state_dict(source_payload)
    model.load_state_dict(source_state, strict=True)
    del source_payload, source_state
    model.to(device)

    # The official ADDT clean-training script initializes EMA from the loaded
    # pre-trained model and applies a power-law EMA warm-up.
    import copy

    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )

    train_loader, eval_loader = create_dataloaders(
        args, data_generator, device.type == "cuda"
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_planned_updates = updates_per_epoch * args.epochs

    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = (
            output_dir / "latest_state.pt"
            if args.resume.lower() == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume state not found: {resume_path}")
        start_epoch, global_step = restore_training_state(
            resume_path,
            model,
            ema_model,
            optimizer,
            data_generator,
            device,
            source_sha256,
        )
        print(
            f"[{utc_now()}] Resumed after epoch {start_epoch}, global step {global_step}",
            flush=True,
        )

    manifest = environment_manifest(
        args, source_path, source_sha256, model, diffusion
    )
    manifest["dataset"] = {
        "name": "CIFAR-100",
        "train_examples": len(train_loader.dataset),
        "test_examples": len(eval_loader.dataset),
        "labels_used_for_training": False,
    }
    manifest["updates_per_epoch"] = updates_per_epoch
    manifest["total_planned_updates"] = total_planned_updates
    manifest_name = "manifest.json" if not args.resume else f"resume_manifest_{int(time.time())}.json"
    atomic_json_dump(manifest, output_dir / manifest_name)

    print(
        f"[{utc_now()}] Strict source load OK | params={manifest['parameter_count']:,} "
        f"| diffusion_steps={diffusion.num_timesteps} | source_sha256={source_sha256}",
        flush=True,
    )
    print(
        f"[{utc_now()}] CIFAR-100 train={len(train_loader.dataset):,}, "
        f"test={len(eval_loader.dataset):,} | micro_batch={args.batch_size} "
        f"| accumulation={args.grad_accum_steps} | effective_batch="
        f"{args.batch_size * args.grad_accum_steps} | updates/epoch={updates_per_epoch}",
        flush=True,
    )

    initial_diag = diagnostic_loss(
        ema_model,
        diffusion,
        eval_loader,
        device,
        args.validation_batches,
        args.seed + 100_000,
    )
    if initial_diag:
        print(
            f"[{utc_now()}] Epoch 0 diagnostic EMA loss={initial_diag['loss']:.6f} "
            f"mse={initial_diag['mse']:.6f} vb={initial_diag['vb']:.6f}",
            flush=True,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    metrics_path = output_dir / "metrics.csv"
    stop_early = False
    last_ema_decay = ema_decay_at_step(
        global_step, args.ema_max_decay, args.ema_inv_gamma, args.ema_power
    )

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals: MutableMapping[str, float] = {
            "loss": 0.0,
            "mse": 0.0,
            "vb": 0.0,
        }
        epoch_samples = 0
        update_window_loss = 0.0
        update_window_count = 0
        number_of_batches = len(train_loader)

        for batch_index, (images, _labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            batch_samples = images.shape[0]
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (batch_samples,), device=device
            )

            # Average exactly over all samples in this accumulation group,
            # including CIFAR-100's smaller final batch.
            group_start_batch = (
                batch_index // args.grad_accum_steps
            ) * args.grad_accum_steps
            group_total_samples = min(
                len(train_loader.dataset) - group_start_batch * args.batch_size,
                args.grad_accum_steps * args.batch_size,
            )
            with torch.cuda.amp.autocast(enabled=args.amp):
                terms = diffusion.training_losses(model, images, timesteps)
                backward_loss = terms["loss"].sum() / group_total_samples
            scaler.scale(backward_loss).backward()

            for key in epoch_totals:
                if key in terms:
                    epoch_totals[key] += terms[key].detach().sum().item()
            batch_loss_sum = terms["loss"].detach().sum().item()
            epoch_samples += batch_samples
            update_window_loss += batch_loss_sum
            update_window_count += batch_samples

            should_update = (
                (batch_index + 1) % args.grad_accum_steps == 0
                or batch_index + 1 == number_of_batches
            )
            if not should_update:
                continue

            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
            else:
                grad_norm = torch.tensor(float("nan"), device=device)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_ema_decay = update_ema(
                ema_model,
                model,
                global_step,
                args.ema_max_decay,
                args.ema_inv_gamma,
                args.ema_power,
            )

            if global_step % args.log_every_updates == 0:
                mean_window_loss = update_window_loss / max(update_window_count, 1)
                print(
                    f"[{utc_now()}] epoch={epoch_number:03d}/{args.epochs:03d} "
                    f"batch={batch_index + 1:04d}/{number_of_batches:04d} "
                    f"step={global_step:06d}/{total_planned_updates:06d} "
                    f"loss={mean_window_loss:.6f} grad_norm={float(grad_norm):.4f} "
                    f"ema={last_ema_decay:.7f}",
                    flush=True,
                )
                update_window_loss = 0.0
                update_window_count = 0

            if args.max_updates > 0 and global_step >= args.max_updates:
                stop_early = True
                break

        epoch_seconds = time.time() - epoch_started
        epoch_means = {
            key: value / max(epoch_samples, 1) for key, value in epoch_totals.items()
        }
        diag: Dict[str, float] = {}
        if (
            args.eval_every_epochs > 0
            and (epoch_number % args.eval_every_epochs == 0 or epoch_number == args.epochs)
        ):
            diag = diagnostic_loss(
                ema_model,
                diffusion,
                eval_loader,
                device,
                args.validation_batches,
                args.seed + 100_000,
            )

        append_metrics(
            metrics_path,
            {
                "utc_time": utc_now(),
                "epoch": epoch_number,
                "global_step": global_step,
                "train_loss": epoch_means["loss"],
                "train_mse": epoch_means["mse"],
                "train_vb": epoch_means["vb"],
                "ema_decay": last_ema_decay,
                "diagnostic_ema_loss": diag.get("loss", ""),
                "diagnostic_ema_mse": diag.get("mse", ""),
                "diagnostic_ema_vb": diag.get("vb", ""),
                "epoch_seconds": epoch_seconds,
            },
        )
        diag_text = (
            f" | diagnostic_ema_loss={diag['loss']:.6f}" if diag else ""
        )
        print(
            f"[{utc_now()}] EPOCH COMPLETE {epoch_number:03d}/{args.epochs:03d} "
            f"train_loss={epoch_means['loss']:.6f} "
            f"time={epoch_seconds / 60:.2f} min{diag_text}",
            flush=True,
        )

        if stop_early:
            smoke_path = output_dir / f"smoke_ema_step_{global_step:06d}.pt"
            atomic_torch_save(cpu_state_dict(ema_model), smoke_path)
            print(f"[{utc_now()}] Smoke-test checkpoint saved: {smoke_path}", flush=True)
            break

        if (
            args.state_every_epochs > 0
            and (
                epoch_number % args.state_every_epochs == 0
                or epoch_number == args.epochs
            )
        ):
            state_path = output_dir / "latest_state.pt"
            save_training_state(
                state_path,
                epoch_number,
                global_step,
                model,
                ema_model,
                optimizer,
                data_generator,
                args,
                source_sha256,
            )
            print(f"[{utc_now()}] Resumable state saved: {state_path}", flush=True)

        if (
            args.snapshot_every_epochs > 0
            and (
                epoch_number % args.snapshot_every_epochs == 0
                or epoch_number == args.epochs
            )
        ):
            snapshot_path = output_dir / f"ema_epoch_{epoch_number:04d}.pt"
            atomic_torch_save(cpu_state_dict(ema_model), snapshot_path)
            print(f"[{utc_now()}] EMA snapshot saved: {snapshot_path}", flush=True)

    if stop_early:
        print(
            f"[{utc_now()}] Stopped after requested smoke-test limit of {global_step} updates.",
            flush=True,
        )
        return

    final_ema_path = output_dir / f"cifar100_uncond_from_cifar10_{args.epochs}ep_ema.pt"
    final_model_path = output_dir / f"cifar100_uncond_from_cifar10_{args.epochs}ep_model.pt"
    atomic_torch_save(cpu_state_dict(ema_model), final_ema_path)
    atomic_torch_save(cpu_state_dict(model), final_model_path)
    final_manifest = {
        "completed_at_utc": utc_now(),
        "epochs_completed": args.epochs,
        "global_step": global_step,
        "ema_checkpoint": str(final_ema_path),
        "ema_checkpoint_sha256": sha256_file(final_ema_path),
        "non_ema_checkpoint": str(final_model_path),
        "non_ema_checkpoint_sha256": sha256_file(final_model_path),
        "source_checkpoint_sha256": source_sha256,
        "load_example": (
            "DiffusionPurificationModel(device, "
            f"checkpoint_path='{final_ema_path}')"
        ),
    }
    atomic_json_dump(final_manifest, output_dir / "completion.json")
    print(f"[{utc_now()}] TRAINING COMPLETE", flush=True)
    print(f"EMA checkpoint: {final_ema_path}", flush=True)
    print(f"EMA SHA256: {final_manifest['ema_checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
