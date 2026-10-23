"""
End-to-End API Integration & Preprocessing Validation Tests
============================================================
Validates:
1. Production checkpoint SHA-256 and loading.
2. Health & Status endpoints (/ and /api/status).
3. Image inference across baseline, face-crop, multi-crop, and diagnostic endpoints.
4. Input validation (invalid file extensions, empty payloads, oversized files, corrupt bytes, micro-dimensions).
5. Out-of-bounds threshold parameter rejection.
6. Explicit rejection of video files (confirming video processing is not implemented).
"""
import hashlib
import io
import os
import sys
from pathlib import Path
from PIL import Image

import pytest
from fastapi.testclient import TestClient

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.api import app
from app.config import MODEL_WEIGHTS_PATH, DEFAULT_PRODUCTION_THRESHOLD, EXPERIMENTAL_FACE_CROP_THRESHOLD

client = TestClient(app)

REQUIRED_CHECKPOINT_SHA256 = "4ff0c6245d67265cdd0feb212a5028d0df0f69959142a41fa270e976da15b887"

def create_synthetic_test_jpeg(width=256, height=256, color=(120, 150, 180)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestEndToEndApiValidation:

    @pytest.mark.model
    def test_production_checkpoint_sha256_integrity(self):
        """Verifies the production checkpoint exists and strictly matches required SHA-256."""
        assert os.path.exists(MODEL_WEIGHTS_PATH), f"Checkpoint missing at {MODEL_WEIGHTS_PATH}"
        h = hashlib.sha256()
        with open(MODEL_WEIGHTS_PATH, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()
        assert actual_sha == REQUIRED_CHECKPOINT_SHA256, (
            f"Checkpoint SHA-256 mismatch: {actual_sha} != {REQUIRED_CHECKPOINT_SHA256}"
        )

    @pytest.mark.model
    def test_status_endpoints(self):
        """Verifies GET / and GET /api/status report service ready and weights loaded."""
        res_root = client.get("/")
        assert res_root.status_code == 200
        root_data = res_root.json()
        assert root_data["status"] == "ready"
        assert root_data["weights_loaded"] is True
        assert root_data["model_architecture"] == "efficientnet_b0"

        res_status = client.get("/api/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["weights_loaded"] is True
        assert status_data["classes"] == ["FAKE", "REAL"]

    @pytest.mark.model
    def test_image_predict_baseline_mode(self):
        """Tests standard baseline full-image inference endpoint."""
        img_bytes = create_synthetic_test_jpeg(300, 300)
        res = client.post(
            "/predict",
            files={"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["prediction"] in ("REAL", "FAKE")
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data
        assert "REAL" in data["probabilities"] and "FAKE" in data["probabilities"]
        assert data["decision"]["threshold_applied"] == DEFAULT_PRODUCTION_THRESHOLD

    @pytest.mark.model
    def test_image_predict_experimental_modes(self):
        """Tests experimental face-crop and multi-crop inference modes."""
        img_bytes = create_synthetic_test_jpeg(320, 240)

        # 1. Face-crop mode
        res_face = client.post(
            "/predict?mode=experimental_face",
            files={"file": ("test_face.jpg", img_bytes, "image/jpeg")}
        )
        assert res_face.status_code == 200
        data_face = res_face.json()
        assert data_face["status"] == "success"
        assert data_face["decision"]["operating_mode"] in ("experimental_face_crop", "experimental_face_crop_fallback")
        assert data_face["decision"]["threshold_applied"] == EXPERIMENTAL_FACE_CROP_THRESHOLD

        # 2. Multi-crop mode
        res_mc = client.post(
            "/predict?mode=experimental_multi_crop&aggregation=mean",
            files={"file": ("test_mc.jpg", img_bytes, "image/jpeg")}
        )
        assert res_mc.status_code == 200
        data_mc = res_mc.json()
        assert data_mc["status"] == "success"
        assert "crops_evaluated" in data_mc
        assert data_mc["crop_count"] >= 1

    @pytest.mark.model
    def test_diagnostic_endpoint(self):
        """Tests multi-view diagnostic comparison endpoint."""
        img_bytes = create_synthetic_test_jpeg(300, 300)
        res = client.post(
            "/predict/diagnostic",
            files={"file": ("test_diag.jpg", img_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()
        assert "baseline_full_image" in data
        assert "experimental_face_crop" in data
        assert "experimental_multi_crop" in data
        assert "comparison" in data

    def test_unloaded_model_service_unavailable(self, monkeypatch):
        """Verifies HTTP 503 is returned when model weights are not loaded."""
        from app.predictor import predictor
        monkeypatch.setattr(predictor, "is_loaded", False)
        img_bytes = create_synthetic_test_jpeg(100, 100)
        res = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        assert res.status_code == 503
        data = res.json()
        assert "detail" in data
        assert "Model weights have not been loaded" in str(data["detail"])

    def test_input_validation_errors(self):
        """Tests rejection of invalid extensions, empty files, oversized files, and corrupt bytes."""
        # 1. Invalid extensions (.txt, .pdf, .exe)
        for bad_ext in [".txt", ".pdf", ".exe"]:
            res = client.post(
                "/predict",
                files={"file": (f"test{bad_ext}", b"some content", "application/octet-stream")}
            )
            assert res.status_code == 400
            assert "Unsupported file format" in res.json()["detail"]

        # 2. Empty file (0 bytes)
        res_empty = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert res_empty.status_code == 400
        assert "empty" in res_empty.json()["detail"].lower()

        # 3. Oversized file (> 30MB)
        oversized = b"0" * (31 * 1024 * 1024)
        res_oversized = client.post(
            "/predict",
            files={"file": ("huge.jpg", oversized, "image/jpeg")}
        )
        assert res_oversized.status_code == 413
        assert "exceeds maximum permitted limit" in res_oversized.json()["detail"]

        # 4. Corrupted bytes
        res_corrupt = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"corrupted payload not an image", "image/jpeg")}
        )
        assert res_corrupt.status_code == 422
        assert "Failed to decode" in res_corrupt.json()["detail"]

        # 5. Micro image (< 10x10)
        micro_img = Image.new("RGB", (4, 4), (200, 100, 50))
        buf = io.BytesIO()
        micro_img.save(buf, format="JPEG")
        res_micro = client.post(
            "/predict",
            files={"file": ("micro.jpg", buf.getvalue(), "image/jpeg")}
        )
        assert res_micro.status_code == 422
        assert "too small" in res_micro.json()["detail"]

    def test_out_of_bounds_parameters_rejected(self):
        """Tests that invalid threshold and aggregation query parameters are rejected with 422."""
        img_bytes = create_synthetic_test_jpeg(100, 100)

        # Threshold > 1.0
        res_thresh = client.post(
            "/predict?threshold=1.5",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert res_thresh.status_code == 422

        # Invalid aggregation parameter
        res_agg = client.post(
            "/predict?mode=experimental_multi_crop&aggregation=unsupported_method",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert res_agg.status_code == 422

    def test_video_file_upload_rejected_as_unsupported(self):
        """
        Validates that video uploads (.mp4, .avi, .mov) are strictly rejected by the backend,
        confirming that video frame extraction and video inference are not implemented in the API.
        """
        for vid_ext in [".mp4", ".avi", ".mov", ".mkv"]:
            res = client.post(
                "/predict",
                files={"file": (f"test_video{vid_ext}", b"\x00\x00\x00 ftypmp42", "video/mp4")}
            )
            assert res.status_code == 400
            assert "Unsupported file format" in res.json()["detail"]
            assert vid_ext in res.json()["detail"]
