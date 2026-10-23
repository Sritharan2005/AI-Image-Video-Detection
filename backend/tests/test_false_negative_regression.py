"""
Regression Tracking Test for Known False-Negative AI Portrait
==============================================================
Test Image: AI-generated portrait of a man working on a laptop in a modern office.
Path: backend/data/test_cases/ai_man_office_laptop.png

Documented Diagnostic Failure across All Pipeline Modes:
1. Baseline Full-Image mode: P(FAKE) = 4.76% -> prediction REAL (threshold 0.50)
2. Experimental Face-Crop mode: face_detected = true, P(FAKE) = 9.76% -> prediction REAL (threshold 0.35)
3. Experimental Multi-Crop (Mean): P(FAKE) = 5.64% -> prediction REAL (threshold 0.50)
4. Experimental Multi-Crop (Max): P(FAKE) = 9.76% -> prediction REAL (threshold 0.50)

Per-crop breakdown:
- global_center : 4.76%
- top (face area): 0.98%
- center        : 5.20%
- bottom (desk) : 7.48%
- face_crop     : 9.76%

This regression test verifies that all pipelines execute deterministically, records
the actual empirical results, and tracks this false-negative failure mode without
fabricating metrics or hardcoding passing outputs.
"""
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from app.predictor import predictor

pytestmark = pytest.mark.model

TEST_IMAGE_PATH = backend_dir / "data" / "test_cases" / "ai_man_office_laptop.png"

def test_known_false_negative_tracking():
    assert TEST_IMAGE_PATH.exists(), f"Tracked test image not found at {TEST_IMAGE_PATH}"

    with open(TEST_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # 1. Baseline Full-Image Evaluation
    baseline_res = predictor.predict_image_bytes(image_bytes, use_face_crop=False, fake_threshold=0.50)
    base_fake_p = baseline_res["probabilities"]["FAKE"]
    base_pred = baseline_res["prediction"]

    # 2. Experimental Face-Crop Evaluation
    face_res = predictor.predict_image_bytes(image_bytes, use_face_crop=True, fake_threshold=0.35)
    face_fake_p = face_res["probabilities"]["FAKE"]
    face_pred = face_res["prediction"]
    face_detected = face_res["face_detected"]

    # 3. Experimental Multi-Crop with MEAN Aggregation
    multi_mean_res = predictor.predict_multi_crop(image_bytes, fake_threshold=0.50, aggregation="mean")
    multi_mean_fake_p = multi_mean_res["probabilities"]["FAKE"]
    multi_mean_pred = multi_mean_res["prediction"]

    # 4. Experimental Multi-Crop with MAX Aggregation
    multi_max_res = predictor.predict_multi_crop(image_bytes, fake_threshold=0.50, aggregation="max")
    multi_max_fake_p = multi_max_res["probabilities"]["FAKE"]
    multi_max_pred = multi_max_res["prediction"]
    crops_eval = multi_max_res["crops_evaluated"]

    print("\n" + "=" * 78)
    print("--- REGRESSION TRACKING: AI MAN OFFICE LAPTOP ---")
    print("=" * 78)
    print(f"Ground Truth          : FAKE (AI-Generated)")
    print(f"1. Baseline Full-Image: P(FAKE)={base_fake_p*100:.2f}%, P(REAL)={baseline_res['probabilities']['REAL']*100:.2f}% -> Pred: {base_pred} (threshold=0.50)")
    print(f"2. Face-Crop Mode     : P(FAKE)={face_fake_p*100:.2f}%, P(REAL)={face_res['probabilities']['REAL']*100:.2f}% -> Pred: {face_pred} (threshold=0.35)")
    print(f"3. Multi-Crop (Mean)  : P(FAKE)={multi_mean_fake_p*100:.2f}%, P(REAL)={multi_mean_res['probabilities']['REAL']*100:.2f}% -> Pred: {multi_mean_pred} (threshold=0.50)")
    print(f"4. Multi-Crop (Max)   : P(FAKE)={multi_max_fake_p*100:.2f}%, P(REAL)={multi_max_res['probabilities']['REAL']*100:.2f}% -> Pred: {multi_max_pred} (threshold=0.50)")
    print(f"Face Localized        : {face_detected} (Bbox: {face_res['face_bbox']})")
    print("-" * 78)
    print("Per-Crop Spatial Breakdown:")
    for c in crops_eval:
        print(f"  Crop {c['crop_name']:<14}: size={c['dimensions'][0]}x{c['dimensions'][1]} | P(FAKE)={c['probabilities']['FAKE']*100:6.2f}% | P(REAL)={c['probabilities']['REAL']*100:6.2f}%")
    print("=" * 78)

    # Assert deterministic output replication across all modes
    assert abs(base_fake_p - 0.0476) < 0.005, f"Unexpected baseline probability drift: {base_fake_p}"
    assert abs(face_fake_p - 0.0976) < 0.005, f"Unexpected face-crop probability drift: {face_fake_p}"
    assert abs(multi_mean_fake_p - 0.0564) < 0.005, f"Unexpected multi-crop mean probability drift: {multi_mean_fake_p}"
    assert abs(multi_max_fake_p - 0.0976) < 0.005, f"Unexpected multi-crop max probability drift: {multi_max_fake_p}"
    assert face_detected is True, "Face localization failed on office portrait"
    assert len(crops_eval) == 5, f"Expected 5 crops for vertical portrait with face, got {len(crops_eval)}"

    # Document active failure: All 4 modes currently misclassify this image as REAL
    assert base_pred == "REAL", "Expected baseline to exhibit documented False Negative"
    assert face_pred == "REAL", "Expected face-crop to exhibit documented False Negative"
    assert multi_mean_pred == "REAL", "Expected multi-crop mean to exhibit documented False Negative"
    assert multi_max_pred == "REAL", "Expected multi-crop max to exhibit documented False Negative"

    print("[TRACKED] Active false-negative confirmed and tracked across all 4 modes for future fine-tuning.")

if __name__ == "__main__":
    test_known_false_negative_tracking()
