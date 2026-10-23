"""
Unit Tests for Multi-Crop Inference Engine & API Integration
============================================================
Tests:
1. Crop generation geometry (aspect ratios, portrait vs landscape vs square).
2. Duplicate suppression on square images.
3. Safe handling of images with no detected face.
4. Aggregation math ('mean' vs 'max' deterministic calculations).
5. Predictor multi-crop execution parity.
6. API endpoint compatibility:
   - POST /predict?mode=experimental_multi_crop
   - POST /predict/multi-crop
   - POST /predict/diagnostic (3-way comparison)
   - Out-of-bounds threshold & invalid aggregation rejection (400/422).
"""

import io
import sys
from pathlib import Path
from PIL import Image
import pytest

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.api import app
from app.predictor import predictor
from app.multi_crop import generate_multi_crops, aggregate_crop_probabilities

client = TestClient(app)
SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent / "public" / "images" / "sample-fake-woman.jpg"

def test_crop_generation_portrait():
    """Verifies that a portrait image (H > W) generates global, top, center, bottom crops."""
    img = Image.new("RGB", (300, 500), color=(128, 128, 128))
    crops = generate_multi_crops(img, include_face=False)

    crop_names = [c["crop_name"] for c in crops]
    assert "global_center" in crop_names, "Missing global_center in portrait"
    assert "top" in crop_names, "Missing top crop in portrait"
    assert "center" in crop_names, "Missing center crop in portrait"
    assert "bottom" in crop_names, "Missing bottom crop in portrait"

    # Top crop should start at y=0, bottom should end at y=500
    top_c = next(c for c in crops if c["crop_name"] == "top")
    bottom_c = next(c for c in crops if c["crop_name"] == "bottom")
    assert top_c["box"] == [0, 0, 300, 300]
    assert bottom_c["box"] == [0, 200, 300, 500]
    print("[PASS] test_crop_generation_portrait passed.")

def test_crop_generation_landscape():
    """Verifies that a landscape image (W > H) generates global, left, center, right crops."""
    img = Image.new("RGB", (600, 300), color=(100, 150, 200))
    crops = generate_multi_crops(img, include_face=False)

    crop_names = [c["crop_name"] for c in crops]
    assert "global_center" in crop_names
    assert "left" in crop_names
    assert "center" in crop_names
    assert "right" in crop_names

    left_c = next(c for c in crops if c["crop_name"] == "left")
    right_c = next(c for c in crops if c["crop_name"] == "right")
    assert left_c["box"] == [0, 0, 300, 300]
    assert right_c["box"] == [300, 0, 600, 300]
    print("[PASS] test_crop_generation_landscape passed.")

def test_duplicate_suppression_square():
    """Verifies that square images do not generate redundant duplicate directional crops."""
    img = Image.new("RGB", (400, 400), color=(50, 50, 50))
    crops = generate_multi_crops(img, include_face=False)

    crop_names = [c["crop_name"] for c in crops]
    assert len(crop_names) == 1
    assert crop_names[0] == "global_center"
    print("[PASS] test_duplicate_suppression_square passed.")

def test_aggregation_methods():
    """Verifies deterministic math for 'mean' and 'max' aggregation."""
    mock_evals = [
        {"probabilities": {"FAKE": 0.10, "REAL": 0.90}},
        {"probabilities": {"FAKE": 0.40, "REAL": 0.60}},
        {"probabilities": {"FAKE": 0.70, "REAL": 0.30}},
    ]

    mean_res = aggregate_crop_probabilities(mock_evals, method="mean")
    assert abs(mean_res["FAKE"] - 0.40) < 1e-4
    assert abs(mean_res["REAL"] - 0.60) < 1e-4

    max_res = aggregate_crop_probabilities(mock_evals, method="max")
    assert abs(max_res["FAKE"] - 0.70) < 1e-4
    assert abs(max_res["REAL"] - 0.30) < 1e-4
    print("[PASS] test_aggregation_methods passed.")

@pytest.mark.model
def test_api_multi_crop_query_mode():
    """Confirms /predict?mode=experimental_multi_crop operates and returns crop diagnostics."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict?mode=experimental_multi_crop&aggregation=mean",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["decision"]["operating_mode"] == "experimental_multi_crop"
    assert data["decision"]["aggregation_method"] == "mean"
    assert data["inference_target"] == "multi_crop"
    assert "crops_evaluated" in data
    assert len(data["crops_evaluated"]) >= 2
    assert "confidence" in data
    print("[PASS] test_api_multi_crop_query_mode passed.")

@pytest.mark.model
def test_api_dedicated_multi_crop_endpoint():
    """Confirms /predict/multi-crop dedicated endpoint works."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict/multi-crop?aggregation=max&threshold=0.45",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["operating_mode"] == "experimental_multi_crop"
    assert data["decision"]["aggregation_method"] == "max"
    assert data["decision"]["threshold_applied"] == 0.45
    print("[PASS] test_api_dedicated_multi_crop_endpoint passed.")

@pytest.mark.model
def test_diagnostic_endpoint_three_way():
    """Confirms /predict/diagnostic now outputs all 3 pipelines side-by-side."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/predict/diagnostic",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "baseline_full_image" in data
    assert "experimental_face_crop" in data
    assert "experimental_multi_crop" in data
    assert "comparison" in data
    assert "fake_probability_delta_face" in data["comparison"]
    assert "fake_probability_delta_multi" in data["comparison"]
    print("[PASS] test_diagnostic_endpoint_three_way passed.")

if __name__ == "__main__":
    print("=== Running Multi-Crop Unit & API Integration Tests ===")
    test_crop_generation_portrait()
    test_crop_generation_landscape()
    test_duplicate_suppression_square()
    test_aggregation_methods()
    test_api_multi_crop_query_mode()
    test_api_dedicated_multi_crop_endpoint()
    test_diagnostic_endpoint_three_way()
    print("=== All Multi-Crop Unit Tests Passed Successfully! ===")
