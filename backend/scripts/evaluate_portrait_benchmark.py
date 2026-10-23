"""
DeepGuard AI - Comprehensive Multi-Mode Benchmark Evaluation Script
===================================================================
Evaluates an independent collection of known REAL and AI-generated (FAKE) images
across four distinct inference configurations:
  1. Baseline Full-Image Pipeline (Default production threshold tau = 0.40)
  2. Experimental Face-Crop Pipeline (Default threshold tau = 0.35)
  3. Experimental Multi-Crop with MEAN Aggregation (Default threshold tau = 0.50)
  4. Experimental Multi-Crop with MAX Aggregation (Default threshold tau = 0.50)

Computes and reports genuine, non-fabricated metrics:
  - Confusion Matrix: True FAKE (TP), False REAL (FN), True REAL (TN), False FAKE (FP)
  - Accuracy
  - Fake Precision
  - Fake Recall
  - Fake F1-Score
  - False-Negative Rate (FNR = FN / (TP + FN))
  - False-Positive Rate (FPR = FP / (TN + FP))
  - Per-image breakdown across all evaluated modes
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.predictor import predictor
from app.config import (
    DEFAULT_PRODUCTION_THRESHOLD,
    EXPERIMENTAL_FACE_CROP_THRESHOLD,
    EXPERIMENTAL_MULTI_CROP_THRESHOLD,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard on Independent Real vs Fake Benchmark across all modes")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(backend_dir / "data" / "validation_eval"),
        help="Path to evaluation dataset directory containing FAKE/ and REAL/ subfolders"
    )
    parser.add_argument(
        "--threshold_baseline",
        type=float,
        default=DEFAULT_PRODUCTION_THRESHOLD,
        help=f"Decision threshold for baseline full-image mode (default: {DEFAULT_PRODUCTION_THRESHOLD})"
    )
    parser.add_argument(
        "--threshold_face",
        type=float,
        default=EXPERIMENTAL_FACE_CROP_THRESHOLD,
        help=f"Decision threshold for experimental face-crop mode (default: {EXPERIMENTAL_FACE_CROP_THRESHOLD})"
    )
    parser.add_argument(
        "--threshold_multi_mean",
        type=float,
        default=EXPERIMENTAL_MULTI_CROP_THRESHOLD,
        help=f"Decision threshold for multi-crop mean aggregation (default: {EXPERIMENTAL_MULTI_CROP_THRESHOLD})"
    )
    parser.add_argument(
        "--threshold_multi_max",
        type=float,
        default=EXPERIMENTAL_MULTI_CROP_THRESHOLD,
        help=f"Decision threshold for multi-crop max aggregation (default: {EXPERIMENTAL_MULTI_CROP_THRESHOLD})"
    )
    parser.add_argument(
        "--include_office_case",
        action="store_true",
        default=True,
        help="Include the confirmed false-negative AI office laptop portrait in the evaluation"
    )
    return parser.parse_args()

def compute_binary_metrics(y_true, y_pred):
    """
    y_true: 0 for FAKE, 1 for REAL
    y_pred: 0 for FAKE, 1 for REAL
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)  # Correctly detected Fake
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)  # Missed Fake (False Negative)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)  # Correctly detected Real
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)  # False Alarm (False Positive)

    total = tp + fn + tn + fp
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "total": total,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fnr": fnr,
        "fpr": fpr,
    }

def print_metrics_table(metrics: dict, title: str, threshold: float):
    print("\n" + "=" * 78)
    print(f"   {title.upper()} (Threshold = {threshold:.2f})")
    print("=" * 78)
    print(f"Total Evaluated Samples : {metrics['total']}")
    print(f"Overall Accuracy        : {metrics['accuracy'] * 100:.2f}%")
    print(f"Fake Precision (PPV)    : {metrics['precision'] * 100:.2f}%")
    print(f"Fake Recall (Sensitivity) : {metrics['recall'] * 100:.2f}%")
    print(f"Fake F1-Score           : {metrics['f1']:.4f}")
    print(f"False Negative Rate(FNR): {metrics['fnr'] * 100:.2f}% (Missed AI Fakes)")
    print(f"False Positive Rate(FPR): {metrics['fpr'] * 100:.2f}% (False Alarms on Real)")
    print("-" * 78)
    print("Confusion Matrix (Ground Truth Rows x Predicted Columns):")
    print(f"  Actual FAKE | Pred FAKE (TP): {metrics['tp']:3d}  |  Pred REAL (FN - Missed): {metrics['fn']:3d}")
    print(f"  Actual REAL | Pred FAKE (FP): {metrics['fp']:3d}  |  Pred REAL (TN)         : {metrics['tn']:3d}")
    print("=" * 78)

def main():
    args = parse_args()
    dataset_path = Path(args.dataset_dir)

    if not dataset_path.exists():
        print(f"[Error] Dataset directory does not exist: {dataset_path}")
        sys.exit(1)

    fake_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    fake_files = []
    real_files = []

    fake_dir = dataset_path / "FAKE"
    real_dir = dataset_path / "REAL"

    if fake_dir.exists():
        for ext in fake_extensions:
            fake_files.extend(list(fake_dir.glob(ext)))
    if real_dir.exists():
        for ext in fake_extensions:
            real_files.extend(list(real_dir.glob(ext)))

    tracked_office_path = backend_dir / "data" / "test_cases" / "ai_man_office_laptop.png"
    if args.include_office_case and tracked_office_path.exists() and tracked_office_path not in fake_files:
        fake_files.append(tracked_office_path)

    fake_files = sorted(list(set(fake_files)))
    real_files = sorted(list(set(real_files)))

    print("\n" + "#" * 84)
    print("   DEEPGUARD INDEPENDENT MULTI-MODE EVALUATION BENCHMARK")
    print("#" * 84)
    print(f"Dataset Location       : {dataset_path}")
    print(f"Loaded FAKE Images     : {len(fake_files)}")
    print(f"Loaded REAL Images     : {len(real_files)}")
    print(f"Total Benchmark Suite  : {len(fake_files) + len(real_files)} images")
    print(f"1. Baseline Full-Image : threshold = {args.threshold_baseline}")
    print(f"2. Face-Crop Mode      : threshold = {args.threshold_face}")
    print(f"3. Multi-Crop (Mean)   : threshold = {args.threshold_multi_mean}")
    print(f"4. Multi-Crop (Max)    : threshold = {args.threshold_multi_max}")

    all_files = [(p, 0, "FAKE") for p in fake_files] + [(p, 1, "REAL") for p in real_files]
    sample_results = []

    for path, y_true, label_str in all_files:
        with open(path, "rb") as f:
            b = f.read()

        # 1. Baseline Full-Image
        res_base = predictor.predict_image_bytes(b, use_face_crop=False, fake_threshold=args.threshold_baseline)
        # 2. Face-Crop
        res_face = predictor.predict_image_bytes(b, use_face_crop=True, fake_threshold=args.threshold_face)
        # 3. Multi-Crop with MEAN aggregation
        res_multi_mean = predictor.predict_multi_crop(b, fake_threshold=args.threshold_multi_mean, aggregation="mean")
        # 4. Multi-Crop with MAX aggregation
        res_multi_max = predictor.predict_multi_crop(b, fake_threshold=args.threshold_multi_max, aggregation="max")

        sample_results.append({
            "filename": path.name,
            "path": str(path),
            "ground_truth": label_str,
            "y_true": y_true,
            # Baseline
            "base_p_fake": res_base["probabilities"]["FAKE"],
            "base_pred": res_base["prediction"],
            "base_y_pred": 0 if res_base["prediction"] == "FAKE" else 1,
            # Face-Crop
            "face_p_fake": res_face["probabilities"]["FAKE"],
            "face_pred": res_face["prediction"],
            "face_y_pred": 0 if res_face["prediction"] == "FAKE" else 1,
            "face_detected": res_face["face_detected"],
            # Multi-Crop (Mean)
            "multi_mean_p_fake": res_multi_mean["probabilities"]["FAKE"],
            "multi_mean_pred": res_multi_mean["prediction"],
            "multi_mean_y_pred": 0 if res_multi_mean["prediction"] == "FAKE" else 1,
            # Multi-Crop (Max)
            "multi_max_p_fake": res_multi_max["probabilities"]["FAKE"],
            "multi_max_pred": res_multi_max["prediction"],
            "multi_max_y_pred": 0 if res_multi_max["prediction"] == "FAKE" else 1,
            "crop_count": res_multi_mean["crop_count"],
        })

    # Compute Metrics for all 4 pipelines
    y_true_all = [s["y_true"] for s in sample_results]
    y_base_pred = [s["base_y_pred"] for s in sample_results]
    y_face_pred = [s["face_y_pred"] for s in sample_results]
    y_multi_mean_pred = [s["multi_mean_y_pred"] for s in sample_results]
    y_multi_max_pred = [s["multi_max_y_pred"] for s in sample_results]

    metrics_base = compute_binary_metrics(y_true_all, y_base_pred)
    metrics_face = compute_binary_metrics(y_true_all, y_face_pred)
    metrics_multi_mean = compute_binary_metrics(y_true_all, y_multi_mean_pred)
    metrics_multi_max = compute_binary_metrics(y_true_all, y_multi_max_pred)

    # Print Formatted Comparison Reports
    print_metrics_table(metrics_base, "1. BASELINE FULL-IMAGE PIPELINE", args.threshold_baseline)
    print_metrics_table(metrics_face, "2. EXPERIMENTAL FACE-CROP PIPELINE", args.threshold_face)
    print_metrics_table(metrics_multi_mean, "3. EXPERIMENTAL MULTI-CROP (MEAN AGGREGATION)", args.threshold_multi_mean)
    print_metrics_table(metrics_multi_max, "4. EXPERIMENTAL MULTI-CROP (MAX AGGREGATION)", args.threshold_multi_max)

    # Summary Comparison Table
    print("\n" + "=" * 105)
    print("   OVERALL PERFORMANCE COMPARISON ACROSS ALL INFERENCE MODES")
    print("=" * 105)
    print(f"{'Pipeline Mode':<30} | {'Threshold':<9} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<9} | {'F1-Score':<9} | {'FNR':<7} | {'FPR':<7}")
    print("-" * 105)
    for title, th, m in [
        ("Baseline (Full-Image)", args.threshold_baseline, metrics_base),
        ("Experimental (Face-Crop)", args.threshold_face, metrics_face),
        ("Experimental (Multi-Crop Mean)", args.threshold_multi_mean, metrics_multi_mean),
        ("Experimental (Multi-Crop Max)", args.threshold_multi_max, metrics_multi_max),
    ]:
        print(f"{title:<30} | {th:9.2f} | {m['accuracy']*100:8.2f}% | {m['precision']*100:9.2f}% | {m['recall']*100:8.2f}% | {m['f1']:9.4f} | {m['fnr']*100:6.2f}% | {m['fpr']*100:6.2f}%")
    print("=" * 105)

    # Detailed Per-Sample Inspection Table
    print("\n" + "=" * 115)
    print("DETAILED PER-SAMPLE PREDICTION TRACE ACROSS MODES")
    print("=" * 115)
    print(f"{'Filename':<26} | {'True':<5} | {'Base P':<8} {'Pred':<5} | {'Face P':<8} {'Pred':<5} | {'M-Mean P':<8} {'Pred':<5} | {'M-Max P':<8} {'Pred':<5}")
    print("-" * 115)
    for s in sample_results:
        print(
            f"{s['filename'][:26]:<26} | "
            f"{s['ground_truth']:<5} | "
            f"{s['base_p_fake']*100:6.2f}% {s['base_pred']:<5} | "
            f"{s['face_p_fake']*100:6.2f}% {s['face_pred']:<5} | "
            f"{s['multi_mean_p_fake']*100:6.2f}% {s['multi_mean_pred']:<5} | "
            f"{s['multi_max_p_fake']*100:6.2f}% {s['multi_max_pred']:<5}"
        )
    print("=" * 115)

    # Export report to JSON
    report_output_path = backend_dir / "data" / "independent_benchmark_report.json"
    export_payload = {
        "dataset_path": str(dataset_path),
        "total_samples": len(sample_results),
        "counts": {
            "fake": len(fake_files),
            "real": len(real_files)
        },
        "baseline_full_image": {
            "threshold": args.threshold_baseline,
            "metrics": metrics_base
        },
        "experimental_face_crop": {
            "threshold": args.threshold_face,
            "metrics": metrics_face
        },
        "experimental_multi_crop_mean": {
            "threshold": args.threshold_multi_mean,
            "metrics": metrics_multi_mean
        },
        "experimental_multi_crop_max": {
            "threshold": args.threshold_multi_max,
            "metrics": metrics_multi_max
        },
        "sample_records": sample_results
    }
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n[REPORT SAVED] Full reproducible benchmark report saved to:\n  {report_output_path}\n")

if __name__ == "__main__":
    main()
