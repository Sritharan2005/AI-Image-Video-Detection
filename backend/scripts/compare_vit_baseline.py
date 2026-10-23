"""
DeepGuard AI — Comparative Evaluation: Baseline EfficientNet-B0 vs Candidate ViT-B/16
=====================================================================================
Evaluates the baseline EfficientNet-B0 production model against a candidate
ViT-B/16 checkpoint on the exact same held-out test split.

Strict Safety & Integrity Rules:
1. Does NOT train, fine-tune, or modify any model weights.
2. Checks and verifies that the production checkpoint SHA-256 remains:
   4ff0c6245d67265cdd0feb212a5028d0df0f69959142a41fa270e976da15b887
   (verified both before AND after evaluation).
3. Performs strict preflight verification: refuses to evaluate if candidate
   weights are missing, corrupt, or structurally incompatible.
4. Evaluates both models live on identical preprocessing and the exact same test images.
   Never fabricates, estimates, or substitutes dummy metrics.
5. Never silently substitutes blank images for corrupted files. Pre-scans and verifies
   every image with PIL; logs and excludes unreadable files, reporting exact final counts.
6. Clearly differentiates between OpenFake benchmark evaluations and secondary
   workplace dataset evaluations (displaying prominent sample size warnings for N <= 150).
7. Computes McNemar's paired statistical test for classification agreement/difference.
8. Keeps known out-of-sample diagnostic portraits strictly separate from benchmark metrics.
9. Saves comprehensive evaluation metrics, confusion matrices, and plots to disk.

Usage:
    # Evaluating on OpenFake held-out test set:
    python backend/scripts/compare_vit_baseline.py --test_dir ./openfake_cache/test --dataset_type openfake --candidate_threshold 0.55

    # Evaluating on secondary workplace dataset:
    python backend/scripts/compare_vit_baseline.py --test_dir backend/data/workplace_dataset/test --dataset_type workplace
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import (
    MODEL_WEIGHTS_PATH,
    CLASS_NAMES,
    IMG_SIZE,
    RESIZE_SIZE,
    IMAGE_NET_MEAN,
    IMAGE_NET_STD,
    DEFAULT_PRODUCTION_THRESHOLD,
)
from app.model import build_efficientnet_b0
from app.transforms import get_inference_transforms

REQUIRED_BASELINE_SHA256 = "4ff0c6245d67265cdd0feb212a5028d0df0f69959142a41fa270e976da15b887"
DEFAULT_CANDIDATE_PATH = backend_root / "models" / "candidate_vit_b16_deepguard.pth"
DEFAULT_RESULTS_DIR = backend_root / "evaluation_results"


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_candidate_vit_b16(num_classes: int = 2) -> nn.Module:
    """Constructs ViT-B/16 architecture with binary classification head."""
    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def verify_checkpoint_compatibility(
    model: nn.Module, state_dict: dict, model_name: str
) -> Tuple[bool, str]:
    """
    Verifies that the provided state dictionary matches the expected model architecture keys and shapes.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    if missing:
        return False, f"[{model_name}] Missing {len(missing)} required layers in checkpoint. Sample: {list(missing)[:5]}"

    # Check key tensor shapes
    for key in model_keys:
        expected_shape = model.state_dict()[key].shape
        actual_shape = state_dict[key].shape
        if expected_shape != actual_shape:
            return False, f"[{model_name}] Layer shape mismatch on '{key}': expected {expected_shape}, got {actual_shape}"

    return True, f"[{model_name}] Structural compatibility verified ({len(model_keys)} layers matched)."


def preflight_check(
    baseline_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> Tuple[bool, str]:
    """
    Validates existence, checksums, and directories before running evaluation.
    Refuses to proceed if candidate checkpoint is missing or corrupt.
    """
    # 1. Baseline check
    if not baseline_path.exists():
        return False, f"Baseline model not found at '{baseline_path}'."

    baseline_sha = compute_sha256(baseline_path)
    if baseline_sha != REQUIRED_BASELINE_SHA256:
        return False, (
            f"Baseline checkpoint SHA-256 mismatch!\n"
            f"  Expected : {REQUIRED_BASELINE_SHA256}\n"
            f"  Actual   : {baseline_sha}\n"
            f"Production checkpoint integrity compromised. Aborting."
        )

    # 2. Candidate check
    if not candidate_path.exists():
        return False, (
            f"Candidate ViT-B/16 checkpoint not found at '{candidate_path}'.\n"
            f"To evaluate a candidate, train ViT-B/16 on Google Colab using "
            f"'DeepGuard_ViT_Training.ipynb' and place the resulting weights file at:\n"
            f"  {candidate_path}\n"
            f"Evaluation halted safely (0 evaluations run, no dummy metrics fabricated)."
        )

    if candidate_path.stat().st_size < 1024 * 1024:
        return False, f"Candidate checkpoint file '{candidate_path}' is suspiciously small (< 1 MB). Likely corrupt or truncated."

    # 3. Test dataset check
    if not test_dir.exists():
        return False, f"Test dataset directory '{test_dir}' does not exist."

    fake_dir = test_dir / "FAKE"
    real_dir = test_dir / "REAL"
    if not fake_dir.exists() or not real_dir.exists():
        return False, f"Test directory must contain 'FAKE/' and 'REAL/' subfolders. Found: {[p.name for p in test_dir.iterdir() if p.is_dir()]}"

    return True, "Preflight checks passed."


class VerifiedImageDataset(Dataset):
    """
    Dataset that explicitly pre-verifies image decodability.
    Never silently substitutes blank images for corrupted files.
    Excludes corrupt files and reports exact sample counts.
    """
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.corrupt_files: List[Tuple[Path, str]] = []
        self.class_to_idx = {"FAKE": 0, "REAL": 1}

        fake_dir = root_dir / "FAKE"
        real_dir = root_dir / "REAL"

        all_candidates = []
        for p in sorted(fake_dir.iterdir()):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                all_candidates.append((p, 0))
        for p in sorted(real_dir.iterdir()):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                all_candidates.append((p, 1))

        # Integrity verification pass
        for path, label in all_candidates:
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    img.convert("RGB")
                self.samples.append((path, label))
            except Exception as e:
                self.corrupt_files.append((path, str(e)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = transforms.ToTensor()(rgb_img)
        return tensor, label


def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: VerifiedImageDataset,
    device: torch.device,
    threshold: float = DEFAULT_PRODUCTION_THRESHOLD,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Runs deterministic evaluation on verified dataset using the specified decision threshold.
    Returns full metrics, confusion matrix, probabilities, and individual predictions.
    """
    model.eval()
    model.to(device)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs_fake: List[float] = []
    latencies: List[float] = []

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # ms

            # Class indexing: Index 0 = FAKE, Index 1 = REAL
            fake_prob = float(probs[0].item())
            pred_class = 0 if fake_prob >= threshold else 1

            y_true.append(int(labels.item()))
            y_pred.append(pred_class)
            y_probs_fake.append(fake_prob)

    # Binary confusion matrix (pos_label = 0 -> FAKE)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = tp + fn + tn + fp
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "model_name": model_name,
        "samples": total,
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision_fake": round(prec, 4),
        "recall_fake": round(rec, 4),
        "f1_fake": round(f1, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "true_fake_tp": tp,
            "false_real_fn": fn,
            "true_real_tn": tn,
            "false_fake_fp": fp,
        },
        "avg_latency_ms": round(avg_latency, 2),
        "raw_predictions": y_pred,
        "raw_targets": y_true,
        "raw_fake_probs": y_probs_fake,
    }


def compute_mcnemar_test(
    y_true: List[int],
    y_pred_base: List[int],
    y_pred_cand: List[int],
) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired binary classifications on the exact same samples.
    b: Baseline correct, Candidate incorrect
    c: Baseline incorrect, Candidate correct
    """
    base_correct = [yt == yp for yt, yp in zip(y_true, y_pred_base)]
    cand_correct = [yt == yp for yt, yp in zip(y_true, y_pred_cand)]

    # Contingency table
    both_correct = sum(1 for b, c in zip(base_correct, cand_correct) if b and c)
    both_wrong = sum(1 for b, c in zip(base_correct, cand_correct) if not b and not c)
    base_only = sum(1 for b, c in zip(base_correct, cand_correct) if b and not c)  # b
    cand_only = sum(1 for b, c in zip(base_correct, cand_correct) if not b and c)  # c

    discordant = base_only + cand_only
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # Edwards continuity correction
        statistic = (abs(cand_only - base_only) - 1.0) ** 2 / discordant
        # Approximate p-value from chi2(1)
        # 1 - erf(sqrt(stat/2)) for 1-dof chi2
        from math import erfc, sqrt
        p_value = float(erfc(sqrt(statistic / 2.0)))

    return {
        "both_correct": both_correct,
        "both_incorrect": both_wrong,
        "baseline_only_correct": base_only,
        "candidate_only_correct": cand_only,
        "discordant_pairs": discordant,
        "mcnemar_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 5),
        "statistically_significant_p05": p_value < 0.05,
    }


def probe_single_image(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates an out-of-sample diagnostic image with the given model."""
    if not image_path.exists():
        return {}
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
        tensor = transform(rgb_img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).squeeze(0)
        return {
            "FAKE": round(float(probs[0].item()), 4),
            "REAL": round(float(probs[1].item()), 4),
        }
    except Exception as e:
        print(f"[WARNING] Could not probe '{image_path}': {e}")
        return {}


def save_confusion_matrix_plot(
    base_cm: Dict[str, int],
    cand_cm: Dict[str, int],
    output_path: Path,
    test_name: str,
):
    """Generates and saves a side-by-side confusion matrix plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        base_matrix = np.array([
            [base_cm["true_real_tn"], base_cm["false_fake_fp"]],
            [base_cm["false_real_fn"], base_cm["true_fake_tp"]],
        ])
        cand_matrix = np.array([
            [cand_cm["true_real_tn"], cand_cm["false_fake_fp"]],
            [cand_cm["false_real_fn"], cand_cm["true_fake_tp"]],
        ])

        labels = ["REAL (1)", "FAKE (0)"]

        for ax, matrix, title in zip(
            axes, [base_matrix, cand_matrix], ["Baseline (EfficientNet-B0)", "Candidate (ViT-B/16)"]
        ):
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                cbar=False,
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Actual Ground Truth")
            ax.set_xlabel("Predicted Class")

        plt.suptitle(f"Confusion Matrices: Baseline vs Candidate ({test_name})", fontsize=14, y=1.02)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix visualization to: {output_path}")
    except Exception as e:
        print(f"[NOTICE] Could not plot confusion matrix: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Baseline EfficientNet-B0 vs Candidate ViT-B/16 on Held-Out Test Set"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        help="Path to held-out test directory containing FAKE/ and REAL/ folders. Required if not using standard defaults.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", "openfake", "workplace", "custom"],
        help="Type of evaluation dataset. 'openfake' indicates the primary held-out benchmark; 'workplace' indicates the secondary out-of-domain probe.",
    )
    parser.add_argument(
        "--baseline_weights",
        type=str,
        default=str(MODEL_WEIGHTS_PATH),
        help="Path to baseline EfficientNet-B0 weights file",
    )
    parser.add_argument(
        "--candidate_weights",
        type=str,
        default=str(DEFAULT_CANDIDATE_PATH),
        help="Path to candidate ViT-B/16 weights file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PRODUCTION_THRESHOLD,
        help="Default decision threshold for FAKE class (default: 0.40)",
    )
    parser.add_argument(
        "--candidate_threshold",
        type=float,
        default=None,
        help="Optional optimal validation threshold selected for candidate ViT (e.g., 0.55). If omitted, evaluates at --threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use ('cuda' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "vit_baseline_comparison.json"),
        help="Path to save comparison JSON summary",
    )
    args = parser.parse_args()

    # Determine test directory and dataset type
    if not args.test_dir:
        # Default to workplace test set if available locally, with clear label
        default_wp = backend_root / "data" / "workplace_dataset" / "test"
        if default_wp.exists():
            test_dir = default_wp
            if args.dataset_type == "auto":
                args.dataset_type = "workplace"
        else:
            print("[ERROR] Please provide --test_dir pointing to the test set.")
            sys.exit(1)
    else:
        test_dir = Path(args.test_dir)
        if args.dataset_type == "auto":
            if "workplace" in str(test_dir).lower():
                args.dataset_type = "workplace"
            elif "openfake" in str(test_dir).lower():
                args.dataset_type = "openfake"
            else:
                args.dataset_type = "custom"

    baseline_path = Path(args.baseline_weights)
    candidate_path = Path(args.candidate_weights)
    out_json_path = Path(args.output_json)

    print("=" * 85)
    print("      DEEPGUARD AI — BASELINE vs CANDIDATE ViT-B/16 MODEL COMPARISON       ")
    print("=" * 85)
    print(f"Dataset Evaluation Scope : {args.dataset_type.upper()}")
    print(f"Test Directory           : {test_dir}")

    # 1. Preflight Checks (Initial baseline SHA-256 check)
    passed, msg = preflight_check(baseline_path, candidate_path, test_dir)
    if not passed:
        print(f"\n[ABORTED] {msg}\n")
        sys.exit(1)

    print(f"\n[PREFLIGHT PASS] Baseline and Candidate checkpoints verified.")
    print(f"Baseline Checkpoint  : {baseline_path} (Initial SHA-256 verified)")
    print(f"Candidate Checkpoint : {candidate_path}")

    # Determine compute device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device    : {device}")

    # 2. Shared Transforms & Dataset Verification (No Silent Blank Substitution)
    transform = get_inference_transforms()
    test_dataset = VerifiedImageDataset(root_dir=test_dir, transform=transform)

    fake_count = sum(1 for _, l in test_dataset.samples if l == 0)
    real_count = sum(1 for _, l in test_dataset.samples if l == 1)
    total_test_samples = len(test_dataset)

    print(f"\n[DATASET INTEGRITY AUDIT]")
    print(f"  Verified Decodable Samples : {total_test_samples} total (FAKE: {fake_count}, REAL: {real_count})")
    print(f"  Corrupt/Unreadable Excluded: {len(test_dataset.corrupt_files)}")
    if test_dataset.corrupt_files:
        for cpath, err in test_dataset.corrupt_files:
            print(f"    [EXCLUDED] {cpath}: {err}")

    if fake_count == 0 or real_count == 0:
        print(f"[ERROR] Incomplete test class split! FAKE: {fake_count}, REAL: {real_count}. Aborting.")
        sys.exit(1)

    print(f"  Class Mapping: {test_dataset.class_to_idx} (0=FAKE, 1=REAL)")
    print(f"  Preprocessing: Resize({RESIZE_SIZE}) -> CenterCrop({IMG_SIZE}), ImageNet normalization")

    # Sample size warning for small evaluation sets
    if args.dataset_type == "workplace" or total_test_samples <= 150:
        print("\n" + "!" * 85)
        print(f"[NOTICE: SECONDARY OUT-OF-DOMAIN SANITY CHECK (N={total_test_samples})]")
        print("This evaluation split represents a small, out-of-domain sanity evaluation.")
        print("Due to high statistical variance on small sample counts, do not treat this")
        print("alone as proof of general broad domain convergence or production superiority.")
        print("Primary benchmark evaluation must be conducted on the 1,500-sample OpenFake test split.")
        print("!" * 85)

    # 3. Load Baseline Model (EfficientNet-B0) with Compatibility Checks
    print("\nLoading Baseline EfficientNet-B0 model...")
    baseline_model = build_efficientnet_b0(pretrained=False, num_classes=2)
    baseline_state = torch.load(baseline_path, map_location=device)
    if isinstance(baseline_state, dict) and "state_dict" in baseline_state:
        baseline_state = baseline_state["state_dict"]

    compat_base, base_msg = verify_checkpoint_compatibility(baseline_model, baseline_state, "EfficientNet-B0")
    if not compat_base:
        print(f"[ERROR] Baseline compatibility failure: {base_msg}")
        sys.exit(1)
    baseline_model.load_state_dict(baseline_state, strict=True)
    print(f"  {base_msg}")

    # 4. Load Candidate Model (ViT-B/16) with Compatibility Checks
    print("Loading Candidate ViT-B/16 model...")
    candidate_model = build_candidate_vit_b16(num_classes=2)
    candidate_state = torch.load(candidate_path, map_location=device)
    if isinstance(candidate_state, dict) and "state_dict" in candidate_state:
        candidate_state = candidate_state["state_dict"]
    elif isinstance(candidate_state, dict) and "model_state_dict" in candidate_state:
        candidate_state = candidate_state["model_state_dict"]

    compat_cand, cand_msg = verify_checkpoint_compatibility(candidate_model, candidate_state, "ViT-B/16")
    if not compat_cand:
        print(f"[ERROR] Candidate compatibility failure: {cand_msg}")
        sys.exit(1)
    candidate_model.load_state_dict(candidate_state, strict=True)
    print(f"  {cand_msg}")

    # 5. Run Live Evaluations on the EXACT Same Test Set
    cand_tau = args.candidate_threshold if args.candidate_threshold is not None else args.threshold

    print(f"\nEvaluating Baseline on {total_test_samples} verified samples (@ tau={args.threshold:.2f})...")
    base_metrics = evaluate_model_on_dataset(
        baseline_model, test_dataset, device, threshold=args.threshold, model_name="EfficientNet-B0"
    )

    print(f"Evaluating Candidate on {total_test_samples} verified samples (@ default tau={args.threshold:.2f})...")
    cand_metrics_default = evaluate_model_on_dataset(
        candidate_model, test_dataset, device, threshold=args.threshold, model_name="ViT-B/16 (Default tau)"
    )

    cand_metrics_opt = None
    if cand_tau != args.threshold:
        print(f"Evaluating Candidate on {total_test_samples} verified samples (@ val-tuned tau={cand_tau:.2f})...")
        cand_metrics_opt = evaluate_model_on_dataset(
            candidate_model, test_dataset, device, threshold=cand_tau, model_name=f"ViT-B/16 (Tuned tau={cand_tau:.2f})"
        )

    # 6. McNemar's Paired Statistical Test
    eval_for_mcnemar = cand_metrics_opt if cand_metrics_opt is not None else cand_metrics_default
    mcnemar_results = compute_mcnemar_test(
        base_metrics["raw_targets"],
        base_metrics["raw_predictions"],
        eval_for_mcnemar["raw_predictions"],
    )

    # 7. Side-by-Side Reporting
    print("\n" + "=" * 85)
    print("                   EMPIRICAL EVALUATION RESULTS COMPARISON                     ")
    print("=" * 85)
    row_fmt = "{:<28} {:<24} {:<24}"
    cand_col_name = f"Candidate (tau={eval_for_mcnemar['threshold']:.2f})"
    print(row_fmt.format("Metric", "Baseline (EfficientNet-B0)", cand_col_name))
    print("-" * 85)
    print(row_fmt.format("Verified Sample Size", f"{base_metrics['samples']} images", f"{eval_for_mcnemar['samples']} images"))
    print(row_fmt.format("Decision Threshold", f"tau = {base_metrics['threshold']:.2f}", f"tau = {eval_for_mcnemar['threshold']:.2f}"))
    print(row_fmt.format("Overall Accuracy", f"{base_metrics['accuracy'] * 100:.2f}%", f"{eval_for_mcnemar['accuracy'] * 100:.2f}%"))
    print(row_fmt.format("FAKE Precision", f"{base_metrics['precision_fake'] * 100:.2f}%", f"{eval_for_mcnemar['precision_fake'] * 100:.2f}%"))
    print(row_fmt.format("FAKE Recall", f"{base_metrics['recall_fake'] * 100:.2f}%", f"{eval_for_mcnemar['recall_fake'] * 100:.2f}%"))
    print(row_fmt.format("False Negative Rate (FNR)", f"{base_metrics['false_negative_rate'] * 100:.2f}%", f"{eval_for_mcnemar['false_negative_rate'] * 100:.2f}%"))
    print(row_fmt.format("False Positive Rate (FPR)", f"{base_metrics['false_positive_rate'] * 100:.2f}%", f"{eval_for_mcnemar['false_positive_rate'] * 100:.2f}%"))
    print(row_fmt.format("FAKE F1-Score", f"{base_metrics['f1_fake']:.4f}", f"{eval_for_mcnemar['f1_fake']:.4f}"))
    print(row_fmt.format("Avg Latency (ms/img)", f"{base_metrics['avg_latency_ms']:.1f} ms", f"{eval_for_mcnemar['avg_latency_ms']:.1f} ms"))
    print("-" * 85)

    print("\nPaired Statistical Analysis (McNemar's Test on Identical Test Samples):")
    print(f"  Both Correct          : {mcnemar_results['both_correct']} images")
    print(f"  Both Incorrect        : {mcnemar_results['both_incorrect']} images")
    print(f"  Baseline Only Correct : {mcnemar_results['baseline_only_correct']} images")
    print(f"  Candidate Only Correct: {mcnemar_results['candidate_only_correct']} images")
    print(f"  Discordant Total (b+c): {mcnemar_results['discordant_pairs']} images")
    print(f"  McNemar Chi2 Statistic: {mcnemar_results['mcnemar_statistic']:.4f} | p-value: {mcnemar_results['p_value']:.5f}")
    if mcnemar_results['statistically_significant_p05']:
        print("  Statistical Verdict   : Difference is statistically significant (p < 0.05).")
    else:
        print("  Statistical Verdict   : Difference is NOT statistically significant at p < 0.05.")

    print("\nConfusion Matrix Breakdown:")
    cm_b = base_metrics["confusion_matrix"]
    cm_c = cand_metrics_default["confusion_matrix"]
    print(f"  Baseline  (tau={args.threshold:.2f}) -> TP: {cm_b['true_fake_tp']:3d} | FN: {cm_b['false_real_fn']:3d} | TN: {cm_b['true_real_tn']:3d} | FP: {cm_b['false_fake_fp']:3d}")
    print(f"  Candidate (tau={args.threshold:.2f}) -> TP: {cm_c['true_fake_tp']:3d} | FN: {cm_c['false_real_fn']:3d} | TN: {cm_c['true_real_tn']:3d} | FP: {cm_c['false_fake_fp']:3d}")
    if cand_metrics_opt is not None:
        cm_o = cand_metrics_opt["confusion_matrix"]
        print(f"  Candidate (tau={cand_tau:.2f}) -> TP: {cm_o['true_fake_tp']:3d} | FN: {cm_o['false_real_fn']:3d} | TN: {cm_o['true_real_tn']:3d} | FP: {cm_o['false_fake_fp']:3d}")
    print("=" * 85)

    # 8. Out-of-Sample Known Portrait Probes (Kept separate from benchmark metrics)
    known_cases = [
        ("AI Office Man (Laptop)", backend_root / "data" / "test_cases" / "ai_man_office_laptop.png"),
        ("AI LinkedIn Profile", Path(r"C:\Users\admin\Downloads\linked in profile image.png")),
    ]

    print("\n--- Out-of-Sample Portrait Diagnostic Probes (Not in Test Split) ---")
    probes_summary = {}
    for name, p in known_cases:
        if p.exists():
            base_p = probe_single_image(baseline_model, p, transform, device)
            cand_p = probe_single_image(candidate_model, p, transform, device)
            base_pred = "FAKE" if base_p.get("FAKE", 0) >= args.threshold else "REAL"
            cand_pred = "FAKE" if cand_p.get("FAKE", 0) >= cand_tau else "REAL"
            print(f"Case: {name}")
            print(f"  Baseline  -> P(FAKE)={base_p.get('FAKE', 0)*100:5.2f}% -> Classified as: {base_pred}")
            print(f"  Candidate -> P(FAKE)={cand_p.get('FAKE', 0)*100:5.2f}% -> Classified as: {cand_pred}")
            probes_summary[name] = {
                "path": str(p),
                "baseline": {"probs": base_p, "prediction": base_pred},
                "candidate": {"probs": cand_p, "prediction": cand_pred},
            }
        else:
            print(f"Case: {name} (File not found at '{p}')")
    print("=" * 85)

    # 9. Verify Baseline Checkpoint SHA-256 AFTER Evaluation
    post_sha = compute_sha256(baseline_path)
    if post_sha != REQUIRED_BASELINE_SHA256:
        print(f"[CRITICAL ERROR] Baseline checkpoint SHA-256 changed after evaluation! Expected {REQUIRED_BASELINE_SHA256}, got {post_sha}")
        sys.exit(1)
    print(f"\n[INTEGRITY PASS] Baseline checkpoint SHA-256 confirmed unchanged after evaluation: {post_sha}")

    # Clean raw predictions for JSON export
    base_export = {k: v for k, v in base_metrics.items() if not k.startswith("raw_")}
    cand_def_export = {k: v for k, v in cand_metrics_default.items() if not k.startswith("raw_")}
    cand_opt_export = {k: v for k, v in cand_metrics_opt.items() if not k.startswith("raw_")} if cand_metrics_opt else None

    # 10. Save JSON summary and confusion matrix plot
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "dataset_type": args.dataset_type,
        "test_directory": str(test_dir),
        "test_sample_count": total_test_samples,
        "corrupt_files_excluded": [str(p) for p, _ in test_dataset.corrupt_files],
        "is_small_sample_size": total_test_samples <= 150,
        "baseline_metrics": base_export,
        "candidate_metrics_default_tau": cand_def_export,
        "candidate_metrics_tuned_tau": cand_opt_export,
        "mcnemar_paired_analysis": mcnemar_results,
        "probes": probes_summary,
        "baseline_sha256_verified": post_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved comparison summary to: {out_json_path}")

    plot_path = out_json_path.parent / "vit_baseline_confusion_matrix.png"
    save_confusion_matrix_plot(cm_b, cm_c, plot_path, test_dir.name)


if __name__ == "__main__":
    main()
