"""
Threshold Sensitivity Evaluation Script
Evaluates FAKE decision thresholds (0.50, 0.45, 0.40, 0.35, 0.30) on the separate
labelled validation dataset (13 REAL, 13 FAKE). Does NOT touch the held-out test set.
Computes: Accuracy, Fake Precision, Fake Recall, False-Negative Rate (FNR), False-Positive Rate (FPR),
and raw confusion matrix counts (TP, FP, TN, FN).
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.predictor import predictor

def run_threshold_sensitivity():
    val_dir = backend_dir / "data" / "validation_eval"
    real_images = sorted(list((val_dir / "REAL").glob("*.jpg")))
    fake_images = sorted(list((val_dir / "FAKE").glob("*.jpg")))

    print(f"Loaded validation set: {len(real_images)} REAL, {len(fake_images)} FAKE (Total N={len(real_images) + len(fake_images)})")

    data_records = []

    # Gather predictions for FAKE images (ground truth: 0)
    for p in fake_images:
        with open(p, "rb") as f:
            b = f.read()
        b_res = predictor.predict_image_bytes(b, use_face_crop=False)
        f_res = predictor.predict_image_bytes(b, use_face_crop=True)
        data_records.append({
            "name": p.name,
            "true_idx": 0,  # FAKE
            "base_fake_p": b_res["probabilities"]["FAKE"],
            "face_fake_p": f_res["probabilities"]["FAKE"],
        })

    # Gather predictions for REAL images (ground truth: 1)
    for p in real_images:
        with open(p, "rb") as f:
            b = f.read()
        b_res = predictor.predict_image_bytes(b, use_face_crop=False)
        f_res = predictor.predict_image_bytes(b, use_face_crop=True)
        data_records.append({
            "name": p.name,
            "true_idx": 1,  # REAL
            "base_fake_p": b_res["probabilities"]["FAKE"],
            "face_fake_p": f_res["probabilities"]["FAKE"],
        })

    thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]

    def print_table(prob_key: str, method_title: str):
        print("\n" + "=" * 105)
        print(f"   THRESHOLD SENSITIVITY ANALYSIS: {method_title.upper()}")
        print("=" * 105)
        print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Precision (Fake)':<16} | {'Recall (Fake)':<13} | {'FNR (Missed Fake)':<17} | {'FPR (False Alarm)':<17} | {'TP':<3} {'FP':<3} {'TN':<3} {'FN':<3}")
        print("-" * 105)

        for th in thresholds:
            y_true = [d["true_idx"] for d in data_records]
            # Predict FAKE (0) if P(FAKE) >= th, else REAL (1)
            y_pred = [0 if d[prob_key] >= th else 1 for d in data_records]

            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

            total = tp + tn + fp + fn
            acc = (tp + tn) / total
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

            print(f"  {th:.2f}     | {acc*100:6.2f}%   |     {prec*100:6.2f}%     |    {rec*100:6.2f}%   |      {fnr*100:6.2f}%       |      {fpr*100:6.2f}%      | {tp:2d}  {fp:2d}  {tn:2d}  {fn:2d}")
        print("-" * 105)

    print_table("base_fake_p", "Baseline (Full-Image)")
    print_table("face_fake_p", "Experimental (Face-Crop with Fallback)")

if __name__ == "__main__":
    run_threshold_sensitivity()
