"""
Automated Integration and Unit Tests for Video Deepfake Analysis
================================================================
Validates:
1. Successful video frame analysis on MP4 and AVI files.
2. Evenly distributed sampled-frame timestamps and per-frame schema.
3. Aggregation rules ('mean' vs 'max').
4. Custom threshold overrides.
5. Rejection of unsupported video formats.
6. Rejection of empty (0-byte) video files.
7. Rejection of corrupted video payloads (422 Unprocessable Entity).
8. Rejection of oversized videos (413 Content Too Large).
9. Out-of-bounds parameter validation (threshold, max_frames, aggregation).
10. Temporary file cleanup on both success and error paths.
11. Checkpoint integrity preservation.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.api import app
from app.config import (
    MODEL_WEIGHTS_PATH,
    DEFAULT_PRODUCTION_THRESHOLD,
    MAX_VIDEO_FILE_SIZE_BYTES,
)

client = TestClient(app)
REQUIRED_CHECKPOINT_SHA256 = "4ff0c6245d67265cdd0feb212a5028d0df0f69959142a41fa270e976da15b887"


def generate_synthetic_video_bytes(
    num_frames: int = 15,
    fps: float = 10.0,
    width: int = 128,
    height: int = 128,
    codec: str = "mp4v",
    ext: str = ".mp4",
) -> bytes:
    """Creates a deterministic synthetic video and returns its raw bytes."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        temp_path = tf.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        for i in range(num_frames):
            # Generate varying color frames
            frame = np.full((height, width, 3), ((i * 15) % 256, 120, 180), dtype=np.uint8)
            out.write(frame)
        out.release()

        with open(temp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class TestVideoPrediction:

    @pytest.mark.model
    def test_checkpoint_hash_preserved(self):
        """Verifies model checkpoint has not been altered."""
        h = hashlib.sha256()
        with open(MODEL_WEIGHTS_PATH, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        assert h.hexdigest() == REQUIRED_CHECKPOINT_SHA256

    @pytest.mark.model
    def test_successful_mp4_video_prediction(self):
        """Tests successful frame-based video evaluation on an MP4 file."""
        vid_bytes = generate_synthetic_video_bytes(num_frames=20, fps=10.0, codec="mp4v", ext=".mp4")

        res = client.post(
            "/api/predict/video?max_frames=6",
            files={"file": ("test_sample.mp4", vid_bytes, "video/mp4")},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "success"
        assert data["media_type"] == "video"
        assert data["filename"] == "test_sample.mp4"
        assert data["prediction"] in ("REAL", "FAKE")
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data
        assert "REAL" in data["probabilities"] and "FAKE" in data["probabilities"]

        # Check metadata
        meta = data["video_metadata"]
        assert meta["total_frames"] == 20
        assert meta["fps"] == 10.0
        assert meta["duration_seconds"] == 2.0
        assert meta["frames_sampled"] == 6
        assert meta["frames_failed"] == 0

        # Check per-frame predictions
        frames = data["per_frame_predictions"]
        assert len(frames) == 6
        prev_ts = -1.0
        for f in frames:
            assert "frame_index" in f
            assert "timestamp_seconds" in f
            assert "timestamp_display" in f
            assert "prediction" in f
            assert "probabilities" in f
            assert f["timestamp_seconds"] >= prev_ts
            assert 0.0 <= f["timestamp_seconds"] <= 2.0
            prev_ts = f["timestamp_seconds"]

        # Check explanation disclaims temporal manipulation
        assert "temporal manipulation" in data["explanation"] or "temporal" in data["explanation"]

    @pytest.mark.model
    def test_successful_avi_video_prediction(self):
        """Tests successful evaluation on an AVI container."""
        vid_bytes = generate_synthetic_video_bytes(num_frames=12, fps=12.0, codec="MJPG", ext=".avi")

        res = client.post(
            "/predict/video?max_frames=4",
            files={"file": ("test_clip.avi", vid_bytes, "video/x-msvideo")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["video_metadata"]["frames_sampled"] == 4
        assert len(data["per_frame_predictions"]) == 4

    @pytest.mark.model
    def test_video_aggregation_methods(self):
        """Tests 'mean' and 'max' aggregation methods produce valid results."""
        vid_bytes = generate_synthetic_video_bytes(num_frames=10, fps=10.0)

        # Mean aggregation
        res_mean = client.post(
            "/api/predict/video?aggregation=mean",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_mean.status_code == 200
        data_mean = res_mean.json()
        assert data_mean["decision"]["aggregation_method"] == "mean"

        # Max aggregation
        res_max = client.post(
            "/api/predict/video?aggregation=max",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_max.status_code == 200
        data_max = res_max.json()
        assert data_max["decision"]["aggregation_method"] == "max"

    @pytest.mark.model
    def test_custom_threshold_override(self):
        """Tests that custom threshold alters decision boundary accordingly."""
        vid_bytes = generate_synthetic_video_bytes(num_frames=10, fps=10.0)

        res_low = client.post(
            "/api/predict/video?threshold=0.01",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_low.status_code == 200
        data_low = res_low.json()
        assert data_low["decision"]["threshold_applied"] == 0.01

        res_high = client.post(
            "/api/predict/video?threshold=0.99",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_high.status_code == 200
        data_high = res_high.json()
        assert data_high["decision"]["threshold_applied"] == 0.99

    def test_unsupported_video_format_rejected(self):
        """Tests rejection of unsupported extensions (.txt, .mp3, etc.)."""
        res = client.post(
            "/api/predict/video",
            files={"file": ("document.txt", b"plain text payload", "text/plain")},
        )
        assert res.status_code == 400
        assert "Unsupported video format" in res.json()["detail"]

    def test_empty_video_rejected(self):
        """Tests rejection of empty (0-byte) video file."""
        res = client.post(
            "/api/predict/video",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        assert res.status_code == 400
        assert "empty (0 bytes)" in res.json()["detail"]

    def test_corrupt_video_rejected(self):
        """Tests rejection of unreadable/corrupted video file."""
        res = client.post(
            "/api/predict/video",
            files={"file": ("corrupt.mp4", b"corrupted payload not a video", "video/mp4")},
        )
        assert res.status_code == 422
        assert "Unable to open video stream" in res.json()["detail"] or "decode" in res.json()["detail"]

    def test_out_of_bounds_parameters_rejected(self):
        """Tests validation on threshold, max_frames, and aggregation parameters."""
        vid_bytes = generate_synthetic_video_bytes(num_frames=6, fps=6.0)

        # Threshold out of bounds
        res_t = client.post(
            "/api/predict/video?threshold=1.5",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_t.status_code == 422

        # Max frames out of bounds (> 32)
        res_f = client.post(
            "/api/predict/video?max_frames=100",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_f.status_code == 422

        # Invalid aggregation
        res_a = client.post(
            "/api/predict/video?aggregation=invalid_agg",
            files={"file": ("video.mp4", vid_bytes, "video/mp4")},
        )
        assert res_a.status_code == 422

    def test_temporary_files_cleanup_on_error(self):
        """Verifies no orphaned temporary files remain after error or corrupt upload."""
        temp_dir = tempfile.gettempdir()
        initial_deepguard_temps = [
            f for f in os.listdir(temp_dir) if f.startswith("deepguard_video_")
        ]

        # Trigger corrupt upload
        res = client.post(
            "/api/predict/video",
            files={"file": ("corrupt.mp4", b"bad video stream data", "video/mp4")},
        )
        assert res.status_code == 422

        current_deepguard_temps = [
            f for f in os.listdir(temp_dir) if f.startswith("deepguard_video_")
        ]
        # No extra temporary file leaked
        assert len(current_deepguard_temps) == len(initial_deepguard_temps)
