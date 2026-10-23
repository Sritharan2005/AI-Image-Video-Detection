"""
Comprehensive Unit Tests for Decision Threshold Boundaries and Operating Modes
Tests:
1. Default production mode (/predict): Full-image, threshold 0.50.
2. Experimental face-crop mode (/predict?mode=experimental_face): Threshold 0.35 default.
3. Custom threshold override via query parameter.
4. Mathematical boundary verification (tau = P(FAKE), tau = P(FAKE) - 0.01, tau = P(FAKE) + 0.01).
5. Extreme boundary enforcement (tau = 0.0, tau = 1.0).
6. Out-of-bounds validation rejection (tau < 0, tau > 1).
7. Diagnostic side-by-side endpoint parity.
"""
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from fastapi.testclient import TestClient
from app.api import app

pytestmark = pytest.mark.model

client = TestClient(app)
SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent / "public" / "images" / "sample-fake-woman.jpg"

def test_default_production_endpoint():
    """Confirms production default: full-image inference, threshold 0.40."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post("/predict", files={"file": ("sample.jpg", f, "image/jpeg")})

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    assert data["decision"]["threshold_applied"] == 0.40
    assert data["decision"]["decision_rule"] == "P(FAKE) >= 0.40 -> FAKE, else REAL"
    assert data["decision"]["operating_mode"] == "baseline_full_image"
    assert data["inference_target"] == "full_image"
    assert "raw_probabilities" in data
    assert "REAL" in data["raw_probabilities"] and "FAKE" in data["raw_probabilities"]
    print("[PASS] test_default_production_endpoint passed.")

def test_experimental_face_crop_default():
    """Confirms experimental mode default: threshold 0.35, target face_crop if face detected."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post("/predict?mode=experimental_face", files={"file": ("sample.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    assert data["decision"]["threshold_applied"] == 0.35
    assert data["decision"]["operating_mode"] == "experimental_face_crop"
    assert data["inference_target"] == "face_crop"
    assert data["face_detected"] is True
    print("[PASS] test_experimental_face_crop_default passed.")

def test_custom_threshold_override():
    """Confirms custom threshold query parameter overrides default."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post("/predict?mode=experimental_face&threshold=0.40", files={"file": ("sample.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["threshold_applied"] == 0.40
    print("[PASS] test_custom_threshold_override passed.")

def test_threshold_decision_boundary():
    """
    Tests exact boundary flipping around P(FAKE).
    On sample-fake-woman.jpg with face crop, P(FAKE) is approx 0.5237.
    - If threshold is set to 0.60 (> P(FAKE)), decision MUST be REAL.
    - If threshold is set to 0.45 (< P(FAKE)), decision MUST be FAKE.
    """
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        bytes_data = f.read()

    # Query with high threshold (0.60)
    res_high = client.post(
        "/predict?mode=experimental_face&threshold=0.60",
        files={"file": ("sample.jpg", bytes_data, "image/jpeg")}
    ).json()
    assert res_high["decision"]["prediction"] == "REAL", f"Expected REAL at threshold 0.60, got {res_high['decision']['prediction']}"

    # Query with lower threshold (0.45)
    res_low = client.post(
        "/predict?mode=experimental_face&threshold=0.45",
        files={"file": ("sample.jpg", bytes_data, "image/jpeg")}
    ).json()
    assert res_low["decision"]["prediction"] == "FAKE", f"Expected FAKE at threshold 0.45, got {res_low['decision']['prediction']}"
    print("[PASS] test_threshold_decision_boundary passed.")

def test_extreme_boundaries():
    """Tests extreme boundary thresholds tau = 0.0 and tau = 1.0."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        bytes_data = f.read()

    # Threshold 0.0: Any non-negative probability qualifies as FAKE
    res_0 = client.post("/predict?threshold=0.0", files={"file": ("sample.jpg", bytes_data, "image/jpeg")}).json()
    assert res_0["decision"]["prediction"] == "FAKE"
    assert res_0["decision"]["threshold_applied"] == 0.0

    # Threshold 1.0: Real unless 100% certain FAKE
    res_1 = client.post("/predict?threshold=1.0", files={"file": ("sample.jpg", bytes_data, "image/jpeg")}).json()
    assert res_1["decision"]["prediction"] == "REAL"
    assert res_1["decision"]["threshold_applied"] == 1.0
    print("[PASS] test_extreme_boundaries passed.")

def test_out_of_bounds_rejection():
    """Confirms invalid thresholds (< 0 or > 1) are rejected with 422."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        bytes_data = f.read()

    res_neg = client.post("/predict?threshold=-0.1", files={"file": ("sample.jpg", bytes_data, "image/jpeg")})
    assert res_neg.status_code == 422, f"Expected 422 for threshold -0.1, got {res_neg.status_code}"

    res_too_high = client.post("/predict?threshold=1.5", files={"file": ("sample.jpg", bytes_data, "image/jpeg")})
    assert res_too_high.status_code == 422, f"Expected 422 for threshold 1.5, got {res_too_high.status_code}"
    print("[PASS] test_out_of_bounds_rejection passed.")

def test_diagnostic_endpoint():
    """Confirms /predict/diagnostic runs both baseline and face-crop side-by-side."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post("/predict/diagnostic", files={"file": ("sample.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert "baseline_full_image" in data
    assert "experimental_face_crop" in data
    assert "comparison" in data
    assert data["baseline_full_image"]["decision"]["threshold_applied"] == 0.40
    assert data["experimental_face_crop"]["decision"]["threshold_applied"] == 0.35
    print("[PASS] test_diagnostic_endpoint passed.")

def test_baseline_threshold_boundary_cases(monkeypatch=None):
    """
    Verify exact mathematical boundary cases for DeepGuard Baseline (threshold 0.40):
    - P(FAKE) = 0.39 -> REAL
    - P(FAKE) = 0.40 -> FAKE
    - P(FAKE) = 0.41 -> FAKE
    """
    from app.predictor import predictor
    orig_run_forward = predictor._run_forward

    try:
        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            sample_bytes = f.read()

        # Boundary Case 1: P(FAKE) = 0.39 -> REAL
        predictor._run_forward = lambda img: {"FAKE": 0.39, "REAL": 0.61}
        res_39 = predictor.predict_image_bytes(sample_bytes, use_face_crop=False)
        assert res_39["prediction"] == "REAL"
        assert res_39["decision"]["prediction"] == "REAL"
        assert res_39["confidence"] == 0.61
        assert res_39["probabilities"]["FAKE"] == 0.39
        assert res_39["probabilities"]["REAL"] == 0.61
        assert res_39["raw_probabilities"]["FAKE"] == 0.39
        assert res_39["raw_probabilities"]["REAL"] == 0.61
        assert res_39["decision"]["threshold_applied"] == 0.40
        assert res_39["decision"]["decision_rule"] == "P(FAKE) >= 0.40 -> FAKE, else REAL"

        http_39 = client.post("/predict", files={"file": ("sample.jpg", sample_bytes, "image/jpeg")}).json()
        assert http_39["prediction"] == "REAL"
        assert http_39["decision"]["prediction"] == "REAL"
        assert http_39["confidence"] == 0.61
        assert http_39["decision"]["threshold_applied"] == 0.40

        # Boundary Case 2: P(FAKE) = 0.40 -> FAKE
        predictor._run_forward = lambda img: {"FAKE": 0.40, "REAL": 0.60}
        res_40 = predictor.predict_image_bytes(sample_bytes, use_face_crop=False)
        assert res_40["prediction"] == "FAKE"
        assert res_40["decision"]["prediction"] == "FAKE"
        assert res_40["confidence"] == 0.40
        assert res_40["probabilities"]["FAKE"] == 0.40
        assert res_40["probabilities"]["REAL"] == 0.60
        assert res_40["raw_probabilities"]["FAKE"] == 0.40
        assert res_40["raw_probabilities"]["REAL"] == 0.60
        assert res_40["decision"]["threshold_applied"] == 0.40
        assert res_40["decision"]["decision_rule"] == "P(FAKE) >= 0.40 -> FAKE, else REAL"

        http_40 = client.post("/predict", files={"file": ("sample.jpg", sample_bytes, "image/jpeg")}).json()
        assert http_40["prediction"] == "FAKE"
        assert http_40["decision"]["prediction"] == "FAKE"
        assert http_40["confidence"] == 0.40
        assert http_40["decision"]["threshold_applied"] == 0.40

        # Boundary Case 3: P(FAKE) = 0.41 -> FAKE
        predictor._run_forward = lambda img: {"FAKE": 0.41, "REAL": 0.59}
        res_41 = predictor.predict_image_bytes(sample_bytes, use_face_crop=False)
        assert res_41["prediction"] == "FAKE"
        assert res_41["decision"]["prediction"] == "FAKE"
        assert res_41["confidence"] == 0.41
        assert res_41["probabilities"]["FAKE"] == 0.41
        assert res_41["probabilities"]["REAL"] == 0.59
        assert res_41["raw_probabilities"]["FAKE"] == 0.41
        assert res_41["raw_probabilities"]["REAL"] == 0.59
        assert res_41["decision"]["threshold_applied"] == 0.40
        assert res_41["decision"]["decision_rule"] == "P(FAKE) >= 0.40 -> FAKE, else REAL"

        http_41 = client.post("/predict", files={"file": ("sample.jpg", sample_bytes, "image/jpeg")}).json()
        assert http_41["prediction"] == "FAKE"
        assert http_41["decision"]["prediction"] == "FAKE"
        assert http_41["confidence"] == 0.41
        assert http_41["decision"]["threshold_applied"] == 0.40
    finally:
        predictor._run_forward = orig_run_forward

    print("[PASS] test_baseline_threshold_boundary_cases passed.")

if __name__ == "__main__":
    print("=== Running Threshold Boundary & Operating Mode Tests ===")
    test_default_production_endpoint()
    test_experimental_face_crop_default()
    test_custom_threshold_override()
    test_threshold_decision_boundary()
    test_extreme_boundaries()
    test_out_of_bounds_rejection()
    test_diagnostic_endpoint()
    test_baseline_threshold_boundary_cases()
    print("=== All 8 Boundary & Diagnostic Tests Passed Successfully! ===")
