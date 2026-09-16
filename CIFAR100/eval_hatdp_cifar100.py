#!/usr/bin/env python3
"""Evaluate the trained CIFAR-100 HAT-DP purifier on reproducible subsets.

Supported attacks:

* AutoAttack Standard, L-infinity, epsilon=8/255;
* AutoAttack Standard, L2, epsilon=0.5;
* adaptive BPDA+EOT, L-infinity, epsilon=8/255.

AutoAttack follows the manuscript's transfer-style protocol: it attacks the
frozen bare MixMo WRN-28-10 and the resulting adversarial images are evaluated
through the actual HAT-DP purifier plus classifier.  BPDA+EOT attacks the full
stochastic pipeline with identity BPDA in the backward pass.

All attacks use the same fixed random 512-image CIFAR-100 test subset within a
seed.  The default seeds are 0, 1, and 2, so the script reports mean and sample
standard deviation over three independently selected subsets/runs.  Every exact
ordered index list is saved and hashed.  BPDA+EOT defaults to the common
DiffPure-style 50 attack steps, 15 attack EOT repetitions, and 150 defense
verification repetitions; all values remain configurable.

Examples:

    python -u eval_hatdp_cifar100.py --mode auto_both
    python -u eval_hatdp_cifar100.py --mode bpda_eot
    python -u eval_hatdp_cifar100.py --mode all --seeds 0 1 2

Each invocation creates a run directory with ``summary.json``, ``metrics.csv``,
``console.log``, and one exact subset-index JSON per seed.  ``summary.json`` is
updated atomically after each batch and contains raw per-seed results plus
mean/sample-standard-deviation aggregates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from autoattack import AutoAttack
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor

from bpda_eot_attack import BPDA_EOT_Attack
import diffusion as diffusion_module
from diffusion import DiffusionPurificationModel
from mixmo_cifar100_wrn import load_mixmo_cifar100_classifier


os.environ.setdefault("OMP_NUM_THREADS", "8")

# ``diffusion.py`` saves a debug PNG on every denoise call.  Evaluation invokes
# denoise hundreds of thousands of times, so disable that diagnostic side effect
# here without altering the purifier computation or the shared training module.
diffusion_module.save_image = lambda *args, **kwargs: None

MODE_AUTO_LINF = "auto_linf"
MODE_AUTO_L2 = "auto_l2"
MODE_BPDA_EOT = "bpda_eot"

CSV_FIELDS = [
    "seed",
    "attack",
    "batch_index",
    "processed",
    "dataset_size",
    "clean_correct",
    "robust_correct",
    "clean_accuracy",
    "robust_accuracy",
    "elapsed_seconds",
    "utc_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-100 HAT-DP evaluation with subset-matched AutoAttack and "
            "adaptive BPDA+EOT."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            MODE_AUTO_LINF,
            MODE_AUTO_L2,
            "auto_both",
            MODE_BPDA_EOT,
            "all",
        ],
        default="auto_both",
        help=(
            "auto_both runs the two manuscript AutoAttack norms; all also "
            "runs BPDA+EOT."
        ),
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        default="checkpoints/cifar100_hatdp_wrn28_10_seed42.pt",
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
        "--output-root", default="results/hatdp_cifar100_attack_evaluation"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional unique directory name; a UTC timestamp is used otherwise.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help=(
            "Independent seeds. Each seed controls subset selection, "
            "AutoAttack, and stochastic purification (default: 0 1 2)."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow torchvision to download CIFAR-100 when it is absent.",
    )

    # Frozen manuscript inference protocol supplied by the author.
    parser.add_argument("--T", type=int, default=240)
    parser.add_argument("--scale", type=float, default=92000.0)
    parser.add_argument("--purify-ensemble", type=int, default=4)
    parser.add_argument(
        "--eval-bs",
        type=int,
        default=256,
        help="Purifier evaluation batch size; default is tuned for a 48 GB GPU.",
    )

    # AutoAttack is intentionally constructed against the bare classifier.
    parser.add_argument(
        "--aa-bs",
        type=int,
        default=512,
        help="AutoAttack batch size for the bare WRN-28-10 on a 48 GB GPU.",
    )
    parser.add_argument("--aa-linf-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--aa-l2-eps", type=float, default=0.5)
    parser.add_argument("--aa-version", default="standard", choices=["standard"])
    parser.add_argument(
        "--save-adversarial",
        action="store_true",
        help="Also save full AutoAttack/BPDA adversarial tensors (large files).",
    )

    # DiffPure-style BPDA+EOT defaults used by most comparison papers.
    parser.add_argument("--bpda-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--bpda-step-size", type=float, default=2.0 / 255.0)
    parser.add_argument("--bpda-steps", type=int, default=50)
    parser.add_argument("--bpda-eot-attack-reps", type=int, default=15)
    parser.add_argument("--bpda-eot-defense-reps", type=int, default=150)
    parser.add_argument(
        "--bpda-inner-bs",
        type=int,
        default=64,
        help=(
            "Images attacked together inside BPDA. With attack EOT=15, the "
            "default 64 produces an effective attack batch of 960 and was "
            "measured on the 48 GB server."
        ),
    )
    parser.add_argument(
        "--bpda-defense-candidate-bs",
        type=int,
        default=12,
        help=(
            "Maximum candidate images per 150-repetition defense-verification "
            "chunk. Default 12 gives an effective batch of 1800 and avoids "
            "late-attack OOM while retaining high 48 GB GPU utilization."
        ),
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=512,
        help=(
            "Fixed random subset size used by every attack and seed. "
            "512 is the default; 0 uses all 10,000 test images."
        ),
    )
    args = parser.parse_args()

    if args.T < 0 or args.scale <= 0 or args.purify_ensemble < 1:
        parser.error("Require T >= 0, scale > 0, and purify-ensemble >= 1")
    if args.eval_bs < 1 or args.aa_bs < 1 or args.num_workers < 0:
        parser.error("Batch sizes must be positive and num-workers nonnegative")
    if args.aa_linf_eps <= 0 or args.aa_l2_eps <= 0:
        parser.error("AutoAttack epsilon values must be positive")
    if args.bpda_eps <= 0 or args.bpda_step_size <= 0:
        parser.error("BPDA epsilon and step size must be positive")
    if min(
        args.bpda_steps,
        args.bpda_eot_attack_reps,
        args.bpda_eot_defense_reps,
        args.bpda_inner_bs,
        args.bpda_defense_candidate_bs,
    ) < 1:
        parser.error("BPDA step/EOT/batch counts must be positive")
    if args.subset_size < 0 or args.subset_size > 10000:
        parser.error("subset-size must be between 0 and 10000")
    if not args.seeds:
        parser.error("At least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    if args.run_name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name
    ):
        parser.error("run-name may contain only letters, digits, dot, dash, underscore")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_jsonable(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


class TeeStream:
    """Write console output to the terminal and an append-only log file."""

    def __init__(self, terminal: Any, log_handle: Any):
        self.terminal = terminal
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_handle.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8")


class BareClassifier(torch.nn.Module):
    """Official MixMo vanilla CIFAR-100 WRN-28-10 on raw [0, 1] images."""

    def __init__(self, checkpoint: Path, device: torch.device):
        super().__init__()
        self.model, self.metadata = load_mixmo_cifar100_classifier(
            str(checkpoint), device
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class OSGDPurifiedClassifier(torch.nn.Module):
    """Actual one-step OSGD purifier followed by the frozen classifier."""

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        timestep: int,
        scale: float,
        ensemble: int,
        classifier_checkpoint: Path,
    ):
        super().__init__()
        self.timestep = timestep
        self.scale = scale
        self.ensemble = ensemble
        self.base_model = BareClassifier(classifier_checkpoint, device)
        self.pure_model = DiffusionPurificationModel(
            device=device,
            guide_type="osgd",
            checkpoint_path=str(checkpoint),
        )

    def purify(self, images: torch.Tensor) -> torch.Tensor:
        normalized = images * 2.0 - 1.0
        purified = self.pure_model.denoise(
            normalized, self.timestep, self.scale
        )
        return torch.clamp((purified + 1.0) / 2.0, 0.0, 1.0)

    def forward(
        self, images: torch.Tensor, mode: str = "purify_and_classify"
    ) -> torch.Tensor:
        if mode == "purify":
            return self.purify(images)
        if mode == "classify":
            return self.base_model(images)
        if mode != "purify_and_classify":
            raise ValueError(f"Unknown mode: {mode}")

        logits = None
        for _ in range(self.ensemble):
            current = self.base_model(self.purify(images))
            logits = current if logits is None else logits + current
        return logits / self.ensemble


class BatchedBPDAEOTAttack(BPDA_EOT_Attack):
    """Mathematically matched BPDA+EOT with batched defense verification.

    The original helper verifies candidate adversarial images one at a time.
    Batching those independent EOT predictions preserves the estimator while
    making practical use of the server's 48 GB GPU.
    """

    def __init__(self, *args: Any, defense_candidate_bs: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.defense_candidate_bs = defense_candidate_bs

    def eot_defense_verification(
        self,
        X_adv: torch.Tensor,
        y: torch.Tensor,
        correct: torch.Tensor,
        defended: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.logical_and(torch.logical_not(correct), defended)
        if not bool(candidates.any()):
            return defended
        updated = defended.clone()
        positions = torch.nonzero(candidates, as_tuple=False).flatten()
        for start in range(0, positions.numel(), self.defense_candidate_bs):
            current_positions = positions[
                start : start + self.defense_candidate_bs
            ]
            verified, _ = self.purify_and_predict(
                X_adv[current_positions],
                y[current_positions],
                self.config["eot_defense_reps"],
                requires_grad=False,
            )
            updated[current_positions] = verified
        return updated


def resolve_modes(mode: str) -> List[str]:
    if mode == "auto_both":
        return [MODE_AUTO_LINF, MODE_AUTO_L2]
    if mode == "all":
        return [MODE_AUTO_LINF, MODE_AUTO_L2, MODE_BPDA_EOT]
    return [mode]


def make_run_directory(args: argparse.Namespace) -> Path:
    checkpoint_tag = Path(args.diffusion_checkpoint).stem
    seeds_tag = "-".join(str(seed) for seed in args.seeds)
    default_name = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        + f"_{checkpoint_tag}_{args.mode}_seeds{seeds_tag}"
    )
    run_name = args.run_name or default_name
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_test_dataset(args: argparse.Namespace) -> torchvision.datasets.CIFAR100:
    return torchvision.datasets.CIFAR100(
        args.data_root,
        train=False,
        download=args.download,
        transform=Compose([ToTensor()]),
    )


def fixed_subset_indices(
    dataset_size: int, subset_size: int, subset_seed: int
) -> List[int]:
    if subset_size == 0 or subset_size == dataset_size:
        return list(range(dataset_size))
    # Match the NumPy RandomState/choice convention used by DiffPure's public
    # CIFAR subset loader.  Keep the sampled order because stochastic evaluation
    # is order-sensitive; the exact ordered list is persisted with the results.
    indices = np.random.RandomState(subset_seed).choice(
        dataset_size, subset_size, replace=False
    )
    return indices.astype(int).tolist()


def make_loader(
    dataset: torch.utils.data.Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def collect_loader(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    images: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for current_images, current_labels in loader:
        images.append(current_images)
        labels.append(current_labels)
    if not images:
        raise RuntimeError("The evaluation dataset is empty")
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def append_metric(path: Path, row: Dict[str, Any]) -> None:
    has_header = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in CSV_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def validate_logits(logits: torch.Tensor, labels: torch.Tensor, name: str) -> None:
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"{name} returned logits {tuple(logits.shape)} for labels "
            f"{tuple(labels.shape)}"
        )
    if logits.shape[1] != 100:
        raise RuntimeError(
            f"{name} returned {logits.shape[1]} classes, expected 100"
        )


def evaluate_clean_and_adversarial_tensors(
    defense: OSGDPurifiedClassifier,
    clean_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    labels: torch.Tensor,
    attack_name: str,
    batch_size: int,
    metrics_path: Path,
    summary: Dict[str, Any],
    summary_path: Path,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    clean_correct = 0
    robust_correct = 0
    total = labels.shape[0]
    started = time.monotonic()

    for batch_index, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        clean = clean_images[start:end].to(device, non_blocking=True)
        adversarial = adversarial_images[start:end].to(device, non_blocking=True)
        current_labels = labels[start:end].to(device, non_blocking=True)
        # The OSGD guide temporarily enables gradients internally. ``no_grad``
        # permits that local override; ``inference_mode`` does not.
        with torch.no_grad():
            clean_logits = defense(clean)
            robust_logits = defense(adversarial)
        validate_logits(clean_logits, current_labels, "purified clean pipeline")
        validate_logits(robust_logits, current_labels, "purified adversarial pipeline")
        clean_correct += int((clean_logits.argmax(1) == current_labels).sum())
        robust_correct += int((robust_logits.argmax(1) == current_labels).sum())
        processed = end
        elapsed = time.monotonic() - started
        row = {
            "seed": seed,
            "attack": attack_name,
            "batch_index": batch_index,
            "processed": processed,
            "dataset_size": total,
            "clean_correct": clean_correct,
            "robust_correct": robust_correct,
            "clean_accuracy": clean_correct / processed,
            "robust_accuracy": robust_correct / processed,
            "elapsed_seconds": elapsed,
            "utc_time": utc_now(),
        }
        append_metric(metrics_path, row)
        summary["progress"] = row
        write_json_atomic(summary, summary_path)
        print(
            f"[{attack_name}] {processed}/{total} | "
            f"clean={row['clean_accuracy']:.4%} | "
            f"robust={row['robust_accuracy']:.4%}",
            flush=True,
        )

    return {
        "attack": attack_name,
        "seed": seed,
        "dataset_size": total,
        "clean_correct": clean_correct,
        "robust_correct": robust_correct,
        "clean_accuracy": clean_correct / total,
        "robust_accuracy": robust_correct / total,
        "elapsed_seconds_evaluation_only": time.monotonic() - started,
    }


def run_autoattack(
    args: argparse.Namespace,
    mode: str,
    dataset: torchvision.datasets.CIFAR100,
    defense: OSGDPurifiedClassifier,
    run_dir: Path,
    metrics_path: Path,
    summary: Dict[str, Any],
    summary_path: Path,
    device: torch.device,
    seed: int,
    indices: Sequence[int],
    indices_path: Path,
) -> Dict[str, Any]:
    loader = make_loader(
        dataset, indices, args.eval_bs, args.num_workers
    )
    clean_cpu, labels_cpu = collect_loader(loader)
    clean = clean_cpu.to(device)
    labels = labels_cpu.to(device)

    norm = "Linf" if mode == MODE_AUTO_LINF else "L2"
    epsilon = args.aa_linf_eps if mode == MODE_AUTO_LINF else args.aa_l2_eps
    print(
        f"Starting AutoAttack {norm}: target=bare classifier, eps={epsilon}, "
        f"n={len(indices)}, seed={seed}, version={args.aa_version}",
        flush=True,
    )
    attack_started = time.monotonic()
    adversary = AutoAttack(
        defense.base_model,
        norm=norm,
        eps=epsilon,
        seed=seed,
        version=args.aa_version,
        verbose=True,
        device=device,
    )
    adversarial = adversary.run_standard_evaluation(
        clean, labels, bs=args.aa_bs
    ).detach()
    attack_seconds = time.monotonic() - attack_started

    max_delta_linf = float((adversarial - clean).abs().amax())
    max_delta_l2 = float(
        (adversarial - clean).flatten(1).norm(p=2, dim=1).amax()
    )
    tolerance = 1e-5
    observed = max_delta_linf if norm == "Linf" else max_delta_l2
    if observed > epsilon + tolerance:
        raise RuntimeError(
            f"AutoAttack {norm} violated its budget: observed={observed}, "
            f"epsilon={epsilon}"
        )

    if args.save_adversarial:
        torch.save(
            {
                "images": adversarial.cpu(),
                "labels": labels_cpu,
                "indices": indices,
                "norm": norm,
                "epsilon": epsilon,
            },
            run_dir / f"adversarial_{mode}_seed{seed}.pt",
        )

    # Decouple stochastic purification evaluation from AutoAttack's random
    # starts so this mode gives the same result when run alone or in a suite.
    seed_everything(seed)
    result = evaluate_clean_and_adversarial_tensors(
        defense=defense,
        clean_images=clean_cpu,
        adversarial_images=adversarial.cpu(),
        labels=labels_cpu,
        attack_name=mode,
        batch_size=args.eval_bs,
        metrics_path=metrics_path,
        summary=summary,
        summary_path=summary_path,
        device=device,
        seed=seed,
    )
    result.update(
        {
            "attack_target": "bare MixMo vanilla CIFAR-100 WRN-28-10 classifier",
            "final_evaluation_target": "actual HAT-DP purifier + classifier",
            "norm": norm,
            "epsilon": epsilon,
            "autoattack_version": args.aa_version,
            "attack_generation_seconds": attack_seconds,
            "max_observed_linf": max_delta_linf,
            "max_observed_l2": max_delta_l2,
            "test_indices_file": str(indices_path),
            "test_indices_sha256": sha256_jsonable(list(indices)),
        }
    )
    del clean, labels, adversarial
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def iter_inner_batches(
    images: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        yield images[start:end], labels[start:end]


def run_bpda_eot(
    args: argparse.Namespace,
    dataset: torchvision.datasets.CIFAR100,
    defense: OSGDPurifiedClassifier,
    run_dir: Path,
    metrics_path: Path,
    summary: Dict[str, Any],
    summary_path: Path,
    device: torch.device,
    seed: int,
    indices: Sequence[int],
    indices_path: Path,
) -> Dict[str, Any]:
    is_full_test_set = len(indices) == len(dataset)
    if is_full_test_set:
        protocol_label = "complete_cifar100_test_set"
    elif len(indices) == 512:
        protocol_label = "common_fixed_random_512_adaptive_attack_protocol"
    else:
        protocol_label = "custom_fixed_random_adaptive_attack_subset"
    print(
        f"Starting BPDA+EOT: n={len(indices)}, seed={seed}, "
        f"protocol={protocol_label}, "
        f"eps={args.bpda_eps}, step={args.bpda_step_size}, "
        f"steps={args.bpda_steps}, attack_eot={args.bpda_eot_attack_reps}, "
        f"defense_eot={args.bpda_eot_defense_reps}",
        flush=True,
    )

    adversary = BatchedBPDAEOTAttack(
        defense,
        adv_eps=args.bpda_eps,
        adv_eta=args.bpda_step_size,
        adv_steps=args.bpda_steps,
        eot_attack_reps=args.bpda_eot_attack_reps,
        eot_defense_reps=args.bpda_eot_defense_reps,
        defense_candidate_bs=args.bpda_defense_candidate_bs,
    )
    loader = make_loader(dataset, indices, args.eval_bs, args.num_workers)
    clean_correct = 0
    robust_correct = 0
    total = len(indices)
    processed = 0
    all_adversarial: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    max_observed_linf = 0.0
    started = time.monotonic()

    for batch_index, (clean_cpu, labels_cpu) in enumerate(loader):
        clean = clean_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)

        adversarial_parts: List[torch.Tensor] = []
        class_paths: List[torch.Tensor] = []
        for inner_images, inner_labels in iter_inner_batches(
            clean, labels, args.bpda_inner_bs
        ):
            class_path, adversarial_cpu = adversary.attack_batch(
                inner_images.contiguous(), inner_labels.contiguous()
            )
            class_paths.append(class_path)
            adversarial_parts.append(adversarial_cpu)
        class_path = torch.cat(class_paths, dim=1)
        adversarial_cpu = torch.cat(adversarial_parts, dim=0)

        max_linf = float((adversarial_cpu - clean_cpu).abs().amax())
        max_observed_linf = max(max_observed_linf, max_linf)
        if max_linf > args.bpda_eps + 1e-5:
            raise RuntimeError(
                f"BPDA+EOT violated L-inf budget: observed={max_linf}, "
                f"epsilon={args.bpda_eps}"
            )

        # Use the BPDA attack's own EOT/verification decision. Re-evaluating the
        # returned images with the four-sample inference ensemble would replace
        # the declared 15/150 protocol with a different stochastic estimate.
        clean_correct += int(class_path[0].sum())
        robust_correct += int(class_path[-1].sum())
        processed += labels.shape[0]
        elapsed = time.monotonic() - started
        row = {
            "seed": seed,
            "attack": MODE_BPDA_EOT,
            "batch_index": batch_index,
            "processed": processed,
            "dataset_size": total,
            "clean_correct": clean_correct,
            "robust_correct": robust_correct,
            "clean_accuracy": clean_correct / processed,
            "robust_accuracy": robust_correct / processed,
            "elapsed_seconds": elapsed,
            "utc_time": utc_now(),
        }
        append_metric(metrics_path, row)
        summary["progress"] = row
        write_json_atomic(summary, summary_path)
        print(
            f"[{MODE_BPDA_EOT}] {processed}/{total} | "
            f"clean={row['clean_accuracy']:.4%} | "
            f"robust={row['robust_accuracy']:.4%}",
            flush=True,
        )

        if args.save_adversarial:
            all_adversarial.append(adversarial_cpu)
            all_labels.append(labels_cpu)
        del clean, labels, adversarial_cpu, adversarial_parts, class_path, class_paths
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.save_adversarial:
        torch.save(
            {
                "images": torch.cat(all_adversarial, dim=0),
                "labels": torch.cat(all_labels, dim=0),
                "indices": indices,
                "norm": "Linf",
                "epsilon": args.bpda_eps,
            },
            run_dir / f"adversarial_bpda_eot_seed{seed}.pt",
        )

    return {
        "attack": MODE_BPDA_EOT,
        "seed": seed,
        "attack_target": "complete HAT-DP purifier + classifier pipeline",
        "backward_approximation": "identity BPDA",
        "norm": "Linf",
        "epsilon": args.bpda_eps,
        "max_observed_linf": max_observed_linf,
        "perturbation_budget_check": "passed for every processed batch",
        "step_size": args.bpda_step_size,
        "steps": args.bpda_steps,
        "eot_attack_repetitions": args.bpda_eot_attack_reps,
        "eot_defense_repetitions": args.bpda_eot_defense_reps,
        "bpda_inner_batch_size": args.bpda_inner_bs,
        "bpda_defense_candidate_batch_size": args.bpda_defense_candidate_bs,
        "effective_attack_eot_batch_size": (
            args.bpda_inner_bs * args.bpda_eot_attack_reps
        ),
        "maximum_effective_defense_eot_batch_size": (
            args.bpda_defense_candidate_bs * args.bpda_eot_defense_reps
        ),
        "attack_start": "clean image (no random start)",
        "protocol": protocol_label,
        "dataset_size": total,
        "test_indices_file": str(indices_path),
        "test_indices_sha256": sha256_jsonable(list(indices)),
        "clean_correct": clean_correct,
        "robust_correct": robust_correct,
        "clean_accuracy": clean_correct / total,
        "robust_accuracy": robust_correct / total,
        "elapsed_seconds": time.monotonic() - started,
    }


def save_subset_indices(
    run_dir: Path,
    dataset_size: int,
    indices: Sequence[int],
    seed: int,
) -> tuple[Path, Dict[str, Any]]:
    is_full_test_set = len(indices) == dataset_size
    payload: Dict[str, Any] = {
        "dataset": "CIFAR-100 test",
        "dataset_size": dataset_size,
        "selected_size": len(indices),
        "selection": (
            "complete_test_set"
            if is_full_test_set
            else "numpy_randomstate_choice_without_replacement"
        ),
        "subset_seed": None if is_full_test_set else seed,
        "ordered_indices": list(indices),
    }
    payload["indices_sha256"] = sha256_jsonable(payload["ordered_indices"])
    path = run_dir / f"subset_indices_seed{seed}.json"
    write_json_atomic(payload, path)
    return path, payload


def update_aggregates(summary: Dict[str, Any], modes: Sequence[str]) -> None:
    per_seed = summary["results"]["per_seed"]
    aggregates: Dict[str, Any] = {}
    for mode in modes:
        completed = [
            per_seed[str(seed)][mode]
            for seed in summary["seeds"]
            if mode in per_seed.get(str(seed), {})
        ]
        if not completed:
            continue
        clean = np.asarray(
            [item["clean_accuracy"] for item in completed], dtype=np.float64
        )
        robust = np.asarray(
            [item["robust_accuracy"] for item in completed], dtype=np.float64
        )
        count = len(completed)
        clean_std = float(clean.std(ddof=1)) if count > 1 else None
        robust_std = float(robust.std(ddof=1)) if count > 1 else None
        aggregates[mode] = {
            "completed_runs": count,
            "requested_runs": len(summary["seeds"]),
            "seeds": [item["seed"] for item in completed],
            "accuracy_unit": "fraction",
            "standard_deviation": "sample standard deviation (ddof=1)",
            "clean_accuracy_mean": float(clean.mean()),
            "clean_accuracy_std": clean_std,
            "robust_accuracy_mean": float(robust.mean()),
            "robust_accuracy_std": robust_std,
            "clean_accuracy_mean_percent": 100.0 * float(clean.mean()),
            "clean_accuracy_std_percent": (
                None if clean_std is None else 100.0 * clean_std
            ),
            "robust_accuracy_mean_percent": 100.0 * float(robust.mean()),
            "robust_accuracy_std_percent": (
                None if robust_std is None else 100.0 * robust_std
            ),
        }
    summary["results"]["aggregate"] = aggregates


def build_summary(
    args: argparse.Namespace,
    checkpoint: Path,
    classifier_checkpoint: Path,
    classifier_metadata: Dict[str, Any],
    run_dir: Path,
    dataset_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "status": "running",
        "started_at_utc": utc_now(),
        "experiment": "CIFAR-100 full HAT-DP attack evaluation",
        "command": [sys.executable, *sys.argv],
        "run_directory": str(run_dir),
        "configuration": vars(args),
        "frozen_inference_protocol": {
            "T": args.T,
            "scale": args.scale,
            "purify_ensemble": args.purify_ensemble,
        },
        "batching_for_48gb_gpu": {
            "evaluation_batch_size": args.eval_bs,
            "autoattack_batch_size": args.aa_bs,
            "bpda_inner_batch_size": args.bpda_inner_bs,
            "bpda_defense_candidate_batch_size": args.bpda_defense_candidate_bs,
            "bpda_effective_attack_eot_batch_size": (
                args.bpda_inner_bs * args.bpda_eot_attack_reps
            ),
            "bpda_maximum_effective_defense_eot_batch_size": (
                args.bpda_defense_candidate_bs * args.bpda_eot_defense_reps
            ),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "classifier": {
            **classifier_metadata,
            "checkpoint_path": str(classifier_checkpoint),
            "checkpoint_size_bytes": classifier_checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(classifier_checkpoint),
            "parameters_frozen": True,
        },
        "dataset": {
            "name": "CIFAR-100 test",
            "available_examples": dataset_size,
            "protocol_for_every_attack": (
                "complete test set"
                if args.subset_size in (0, dataset_size)
                else f"fixed random subset of {args.subset_size} images per seed"
            ),
            "comparability_requirement": (
                "Within each seed, AutoAttack Linf, AutoAttack L2, and BPDA+EOT "
                "use the identical ordered test-index list. Reuse these saved "
                "lists for every comparison checkpoint."
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "seeds": list(args.seeds),
        "aggregation": {
            "unit": "one independently seeded fixed test subset/run",
            "reported_statistics": "mean and sample standard deviation",
            "ddof": 1,
        },
        "subsets": {},
        "results": {"per_seed": {}, "aggregate": {}},
        "progress": None,
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.diffusion_checkpoint).expanduser().resolve()
    classifier_checkpoint = Path(args.classifier_checkpoint).expanduser().resolve()
    for label, path in (
        ("diffusion checkpoint", checkpoint),
        ("classifier checkpoint", classifier_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    run_dir = make_run_directory(args)
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    console_path = run_dir / "console.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with console_path.open("a", encoding="utf-8", buffering=1) as console_handle:
        sys.stdout = TeeStream(original_stdout, console_handle)
        sys.stderr = TeeStream(original_stderr, console_handle)
        summary: Dict[str, Any] | None = None
        try:
            seed_everything(args.seeds[0])
            dataset = build_test_dataset(args)
            if len(dataset) != 10000:
                raise RuntimeError(
                    f"Expected 10,000 CIFAR-100 test images, found {len(dataset)}"
                )
            print(f"Run directory: {run_dir}", flush=True)
            print(f"Checkpoint: {checkpoint}", flush=True)
            print(
                f"Protocol: T={args.T}, scale={args.scale}, "
                f"ensemble={args.purify_ensemble}, seeds={args.seeds}, "
                f"subset_size={args.subset_size}",
                flush=True,
            )
            defense = OSGDPurifiedClassifier(
                checkpoint=checkpoint,
                device=device,
                timestep=args.T,
                scale=args.scale,
                ensemble=args.purify_ensemble,
                classifier_checkpoint=classifier_checkpoint,
            ).to(device).eval()
            defense.requires_grad_(False)

            # Fail before expensive attacks if model/data/class dimensions mismatch.
            sanity_loader = make_loader(dataset, [0, 1], 2, 0)
            sanity_images, sanity_labels = next(iter(sanity_loader))
            with torch.inference_mode():
                bare_logits = defense.base_model(sanity_images.to(device))
            validate_logits(bare_logits, sanity_labels.to(device), "bare classifier")
            print("Sanity check passed: classifier returns 100 logits.", flush=True)

            summary = build_summary(
                args,
                checkpoint,
                classifier_checkpoint,
                defense.base_model.metadata,
                run_dir,
                len(dataset),
                device,
            )
            modes = resolve_modes(args.mode)
            subsets: Dict[int, tuple[List[int], Path]] = {}
            for seed in args.seeds:
                indices = fixed_subset_indices(
                    len(dataset), args.subset_size, seed
                )
                indices_path, subset_payload = save_subset_indices(
                    run_dir, len(dataset), indices, seed
                )
                subsets[seed] = (indices, indices_path)
                summary["subsets"][str(seed)] = {
                    **{key: value for key, value in subset_payload.items()
                       if key != "ordered_indices"},
                    "indices_file": str(indices_path),
                }
                summary["results"]["per_seed"][str(seed)] = {}
            write_json_atomic(summary, summary_path)

            for seed in args.seeds:
                indices, indices_path = subsets[seed]
                for mode in modes:
                    # Reset before every attack for mode-independent reproducibility.
                    seed_everything(seed)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                    summary["active_seed"] = seed
                    summary["active_attack"] = mode
                    summary["progress"] = None
                    write_json_atomic(summary, summary_path)
                    if mode in (MODE_AUTO_LINF, MODE_AUTO_L2):
                        result = run_autoattack(
                            args,
                            mode,
                            dataset,
                            defense,
                            run_dir,
                            metrics_path,
                            summary,
                            summary_path,
                            device,
                            seed,
                            indices,
                            indices_path,
                        )
                    elif mode == MODE_BPDA_EOT:
                        result = run_bpda_eot(
                            args,
                            dataset,
                            defense,
                            run_dir,
                            metrics_path,
                            summary,
                            summary_path,
                            device,
                            seed,
                            indices,
                            indices_path,
                        )
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")
                    if device.type == "cuda":
                        result["peak_cuda_memory_allocated_gib"] = (
                            torch.cuda.max_memory_allocated(device) / (1024**3)
                        )
                        result["peak_cuda_memory_reserved_gib"] = (
                            torch.cuda.max_memory_reserved(device) / (1024**3)
                        )
                    summary["results"]["per_seed"][str(seed)][mode] = result
                    update_aggregates(summary, modes)
                    summary["progress"] = None
                    write_json_atomic(summary, summary_path)
                    print(
                        f"Completed seed={seed} {mode}: "
                        f"clean={result['clean_accuracy']:.4%}, "
                        f"robust={result['robust_accuracy']:.4%}",
                        flush=True,
                    )

            summary.update(
                {
                    "status": "complete",
                    "completed_at_utc": utc_now(),
                    "active_seed": None,
                    "active_attack": None,
                    "progress": None,
                }
            )
            write_json_atomic(summary, summary_path)
            print(f"All requested attacks completed. Results: {summary_path}")
        except BaseException as error:
            if summary is None:
                summary = {
                    "status": "failed",
                    "started_at_utc": utc_now(),
                    "configuration": vars(args),
                }
            summary.update(
                {
                    "status": "failed",
                    "failed_at_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            write_json_atomic(summary, summary_path)
            print(summary["traceback"], file=sys.stderr, flush=True)
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
