"""
Evaluation script comparing Baseline Full-Image vs. Experimental Face-Crop Inference
on a balanced labelled validation set. Does NOT touch the held-out test set.
Computes: Accuracy, Fake Precision, Fake Recall, False-Negative Rate (FNR), and Confusion Matrices.
"""
import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from app.predictor import predictor

def evaluate_dataset(val_dir: Path):
    real_dir = val_dir / "REAL"
    fake_dir = val_dir / "FAKE"

    real_images = sorted(list(real_dir.glob("*.jpg")))
    fake_images = sorted(list(fake_dir.glob("*.jpg")))

    print(f"Validation Dataset: {len(real_images)} REAL images, {len(fake_images)} FAKE images (Total: {len(real_images) + len(fake_images)})")

    # In our class mapping: 0 = FAKE, 1 = REAL
    # For sklearn binary metrics, let's treat FAKE as the positive class (pos_label = 0)
    fake_class_idx = 0
    real_class_idx = 1

    y_true = []
    # 0 = FAKE, 1 = REAL
    for _ in fake_images:
        y_true.append(fake_class_idx)
    for _ in real_images:
        y_true.append(real_class_idx)

    y_pred_baseline = []
    y_pred_face_crop = []
    face_detected_count = 0
    all_images = fake_images + real_images

    print("\nRunning inference passes...")
    for idx, img_path in enumerate(all_images):
        with open(img_path, "rb") as f:
            data = f.read()

        # 1. Baseline Full Image
        base_res = predictor.predict_image_bytes(data, use_face_crop=False)
        base_idx = fake_class_idx if base_res["prediction"] == "FAKE" else real_class_idx
        y_pred_baseline.append(base_idx)

        # 2. Experimental Face Crop
        face_res = predictor.predict_image_bytes(data, use_face_crop=True)
        face_idx = fake_class_idx if face_res["prediction"] == "FAKE" else real_class_idx
        y_pred_face_crop.append(face_idx)

        if face_res["face_detected"]:
            face_detected_count += 1

        actual_str = "FAKE" if y_true[idx] == fake_class_idx else "REAL"
        print(f"[{idx+1:02d}/{len(all_images):02d}] {img_path.name[:25]:<25} | Actual: {actual_str:4} | Baseline: {base_res['prediction']:4} ({base_res['confidence']*100:.1f}%) | Face-Crop: {face_res['prediction']:4} ({face_res['confidence']*100:.1f}%, Target: {face_res['inference_target']})")

    def compute_metrics(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=fake_class_idx, zero_division=0
        )
        fnr = 1.0 - rec
        # cm rows: True [REAL, FAKE], cols: Pred [REAL, FAKE]
        cm = confusion_matrix(y_true, y_pred, labels=[real_class_idx, fake_class_idx])
        return {
            "name": name,
            "accuracy": acc,
            "precision_fake": prec,
            "recall_fake": rec,
            "fnr_fake": fnr,
            "f1_fake": f1,
            "cm": cm,
        }

    m_base = compute_metrics(y_true, y_pred_baseline, "Baseline (Full Image)")
    m_face = compute_metrics(y_true, y_pred_face_crop, "Experimental (Face-Crop with Fallback)")

    print("\n" + "=" * 80)
    print("      EXPERIMENTAL VALIDATION EVALUATION: BASELINE VS. FACE-CROP")
    print("=" * 80)
    header = f"{'Metric':<32} {'Baseline (Full Image)':<24} {'Experimental (Face-Crop)':<24}"
    print(header)
    print("-" * 80)
    print(f"{'Overall Accuracy':<32} {m_base['accuracy']*100:>20.2f}% {m_face['accuracy']*100:>22.2f}%")
    print(f"{'Fake-Class Precision':<32} {m_base['precision_fake']*100:>20.2f}% {m_face['precision_fake']*100:>22.2f}%")
    print(f"{'Fake-Class Recall':<32} {m_base['recall_fake']*100:>20.2f}% {m_face['recall_fake']*100:>22.2f}%")
    print(f"{'Fake False-Negative Rate (FNR)':<32} {m_base['fnr_fake']*100:>20.2f}% {m_face['fnr_fake']*100:>22.2f}%")
    print(f"{'Fake-Class F1-Score':<32} {m_base['f1_fake']:>21.4f} {m_face['f1_fake']:>23.4f}")
    print("-" * 80)
    print(f"Face Detection Coverage: {face_detected_count}/{len(all_images)} images ({face_detected_count/len(all_images)*100:.1f}%)")
    print("\nConfusion Matrices [Rows = Actual (REAL, FAKE), Cols = Predicted (REAL, FAKE)]:")
    print("--- Baseline Full Image ---")
    print(f"  Actual REAL (N={len(real_images)}):  True REAL = {m_base['cm'][0,0]:2d}  |  False FAKE = {m_base['cm'][0,1]:2d}")
    print(f"  Actual FAKE (N={len(fake_images)}):  False REAL = {m_base['cm'][1,0]:2d}  |  True FAKE  = {m_base['cm'][1,1]:2d}")
    print("--- Experimental Face-Crop ---")
    print(f"  Actual REAL (N={len(real_images)}):  True REAL = {m_face['cm'][0,0]:2d}  |  False FAKE = {m_face['cm'][0,1]:2d}")
    print(f"  Actual FAKE (N={len(fake_images)}):  False REAL = {m_face['cm'][1,0]:2d}  |  True FAKE  = {m_face['cm'][1,1]:2d}")
    print("=" * 80)

if __name__ == "__main__":
    val_path = backend_dir / "data" / "validation_eval"
    evaluate_dataset(val_path)
