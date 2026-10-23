"""
DeepGuard AI - Comparative Evaluation on Untouched Held-Out Test Set
=====================================================================
Compares the original OpenFake checkpoint against the fine-tuned workplace
checkpoint on exactly the same untouched held-out test images.

Reports:
  - Confusion Matrix (TP, FN, TN, FP)
  - Overall Accuracy
  - Fake Precision
  - Fake Recall
  - Fake F1-Score
  - False Negative Rate (FNR)
  - False Positive Rate (FPR)
  - Per-image side-by-side probability and decision comparison

Usage:
    python backend/scripts/compare_models_untouched_test.py --test_dir backend/data/workplace_dataset/test
"""

import argparse
import json
import os
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import (
    MODEL_WEIGHTS_PATH,
    NUM_CLASSES,
    CLASS_NAMES,
    DEFAULT_PRODUCTION_THRESHOLD,
)
from app.model import build_efficientnet_b0
from app.transforms import get_inference_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Original vs Fine-Tuned Checkpoints on Untouched Test Split")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing held-out test FAKE/ and REAL/ subfolders")
    parser.add_argument("--original_weights", type=str, default=str(MODEL_WEIGHTS_PATH), help="Path to original model weights")
    parser.add_argument("--finetuned_weights", type=str, default=str(backend_root / "models" / "efficientnet_b0_workplace_finetuned.pth"), help="Path to fine-tuned model weights")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PRODUCTION_THRESHOLD, help="Decision threshold for FAKE class (default 0.40)")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()

def compute_metrics(y_true, y_pred):
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

    return {
        "total": total,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "fnr": fnr, "fpr": fpr
    }

def print_model_report(title: str, metrics: dict, th: float):
    print("\n" + "=" * 76)
    print(f"   {title.upper()} (Threshold = {th:.2f})")
    print("=" * 76)
    print(f"Total Test Samples     : {metrics['total']}")
    print(f"Overall Accuracy       : {metrics['accuracy'] * 100:.2f}%")
    print(f"Fake Precision (PPV)   : {metrics['precision'] * 100:.2f}%")
    print(f"Fake Recall (Sens.)    : {metrics['recall'] * 100:.2f}%")
    print(f"Fake F1-Score          : {metrics['f1']:.4f}")
    print(f"False Negative Rate(FNR): {metrics['fnr'] * 100:.2f}%")
    print(f"False Positive Rate(FPR): {metrics['fpr'] * 100:.2f}%")
    print("-" * 76)
    print("Confusion Matrix:")
    print(f"  Actual FAKE | Pred FAKE (TP): {metrics['tp']:3d}  |  Pred REAL (FN - Missed): {metrics['fn']:3d}")
    print(f"  Actual REAL | Pred FAKE (FP): {metrics['fp']:3d}  |  Pred REAL (TN)         : {metrics['tn']:3d}")
    print("=" * 76)

def main():
    args = parse_args()

    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from PIL import Image

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    test_path = Path(args.test_dir)

    if not test_path.exists():
        print(f"[Error] Test directory does not exist: {test_path}")
        sys.exit(1)

    if not os.path.exists(args.original_weights):
        print(f"[Error] Original weights not found: {args.original_weights}")
        sys.exit(1)

    if not os.path.exists(args.finetuned_weights):
        print(f"[Error] Fine-tuned weights not found: {args.finetuned_weights}")
        sys.exit(1)

    # Load dataset
    transform = get_inference_transforms()
    test_dataset = ImageFolder(root=str(test_path), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    fake_idx = test_dataset.class_to_idx.get("FAKE", 0)
    real_idx = test_dataset.class_to_idx.get("REAL", 1)

    # Load Model A (Original)
    model_orig = build_efficientnet_b0(pretrained=False, num_classes=NUM_CLASSES)
    sd_orig = torch.load(args.original_weights, map_location=device)
    if isinstance(sd_orig, dict) and "state_dict" in sd_orig:
        sd_orig = sd_orig["state_dict"]
    model_orig.load_state_dict(sd_orig)
    model_orig.to(device).eval()

    # Load Model B (Fine-Tuned)
    model_ft = build_efficientnet_b0(pretrained=False, num_classes=NUM_CLASSES)
    sd_ft = torch.load(args.finetuned_weights, map_location=device)
    if isinstance(sd_ft, dict) and "state_dict" in sd_ft:
        sd_ft = sd_ft["state_dict"]
    model_ft.load_state_dict(sd_ft)
    model_ft.to(device).eval()

    y_true = []
    orig_probs_fake = []
    ft_probs_fake = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            out_orig = model_orig(images)
            out_ft = model_ft(images)

            p_orig = torch.softmax(out_orig, dim=1)[:, fake_idx]
            p_ft = torch.softmax(out_ft, dim=1)[:, fake_idx]

            y_true.extend(labels.numpy().tolist())
            orig_probs_fake.extend(p_orig.cpu().numpy().tolist())
            ft_probs_fake.extend(p_ft.cpu().numpy().tolist())

    # Calculate Decisions
    orig_preds = [0 if p >= args.threshold else 1 for p in orig_probs_fake]
    ft_preds = [0 if p >= args.threshold else 1 for p in ft_probs_fake]

    m_orig = compute_metrics(y_true, orig_preds)
    m_ft = compute_metrics(y_true, ft_preds)

    print("\n" + "#" * 80)
    print("   UNTOUCHED TEST SET HEAD-TO-HEAD BENCHMARK EVALUATION")
    print("#" * 80)
    print_model_report("1. ORIGINAL MODEL (OpenFake Scale-5K Checkpoint)", m_orig, args.threshold)
    print_model_report("2. FINE-TUNED MODEL (Domain-Specific Workplace Checkpoint)", m_ft, args.threshold)

    # Comparative Summary
    print("\n" + "=" * 92)
    print("   HEAD-TO-HEAD COMPARATIVE SUMMARY (Untouched Test Split)")
    print("=" * 92)
    print(f"{'Metric':<26} | {'Original Checkpoint':<26} | {'Fine-Tuned Checkpoint':<26}")
    print("-" * 92)
    print(f"{'Overall Accuracy':<26} | {m_orig['accuracy']*100:6.2f}%                    | {m_ft['accuracy']*100:6.2f}%")
    print(f"{'Fake Precision':<26} | {m_orig['precision']*100:6.2f}%                    | {m_ft['precision']*100:6.2f}%")
    print(f"{'Fake Recall':<26} | {m_orig['recall']*100:6.2f}%                    | {m_ft['recall']*100:6.2f}%")
    print(f"{'Fake F1-Score':<26} | {m_orig['f1']:6.4f}                    | {m_ft['f1']:6.4f}")
    print(f"{'False Negative Rate (FNR)':<26} | {m_orig['fnr']*100:6.2f}%                    | {m_ft['fnr']*100:6.2f}%")
    print(f"{'False Positive Rate (FPR)':<26} | {m_orig['fpr']*100:6.2f}%                    | {m_ft['fpr']*100:6.2f}%")
    print("=" * 92)

if __name__ == "__main__":
    main()
