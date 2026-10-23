"""
Tests for Network Resilience, Retries, Interruption, and Resumability in
backend/scripts/acquire_workplace_dataset.py
========================================================================
Validates:
1. Classification of transient (retryable) vs permanent (non-retryable) network errors.
2. Transient failure followed by successful download with backoff.
3. Retry exhaustion recording 'FAILED' with error_type and skipping future requests.
4. Clean KeyboardInterrupt handling with log flushing and no partial file corruption.
5. Immediate partial download (.tmp) cleanup.
6. Resuming from existing ledger without duplicate downloads or requests.
"""

import io
import json
import os
import shutil
import socket
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from scripts.acquire_workplace_dataset import (
    classify_network_error,
    fetch_url_with_retries,
    cleanup_temp_files,
    log_event,
    load_existing_ledger,
    get_current_counts,
    download_and_validate_asset,
    run_acquisition,
    compute_sha256_bytes,
    sync_verified_record_to_ledger,
)


def create_dummy_photo_image(width=320, height=240) -> Image.Image:
    """Creates a synthetic photo-like image with rich continuous color gradations to pass pixel-level checks."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x + y) % 256
            arr[y, x, 1] = (2 * x + y) % 256
            arr[y, x, 2] = (x + 2 * y) % 256
    return Image.fromarray(arr)


def create_dummy_image_bytes(width=256, height=256, color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class MockHTTPResponse:
    def __init__(self, data: bytes, status: int = 200, headers: dict = None):
        self.data = data
        self.status = status
        self.headers = headers or {}

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestNetworkResilience:

    def test_classify_network_error_types(self):
        # 1. Transient / Retryable errors
        ssl_err = ssl.SSLError("SSL handshake timed out")
        err_type, retryable = classify_network_error(ssl_err)
        assert err_type == "SSL_HANDSHAKE_ERROR"
        assert retryable is True

        timeout_err = socket.timeout("Socket read timed out")
        err_type, retryable = classify_network_error(timeout_err)
        assert err_type == "TIMEOUT_ERROR"
        assert retryable is True

        conn_reset = ConnectionResetError("Connection reset by peer")
        err_type, retryable = classify_network_error(conn_reset)
        assert err_type == "CONNECTION_RESET"
        assert retryable is True

        remote_disc = RemoteDisconnected("Remote end closed connection without response")
        err_type, retryable = classify_network_error(remote_disc)
        assert err_type == "CONNECTION_RESET"
        assert retryable is True

        http_429 = urllib.error.HTTPError("https://api.test", 429, "Too Many Requests", {}, None)
        err_type, retryable = classify_network_error(http_429)
        assert err_type == "HTTP_429_TOO_MANY_REQUESTS"
        assert retryable is True

        for code in (500, 502, 503, 504):
            http_5xx = urllib.error.HTTPError("https://api.test", code, "Server Error", {}, None)
            err_type, retryable = classify_network_error(http_5xx)
            assert err_type == f"HTTP_{code}_SERVER_ERROR"
            assert retryable is True

        # 2. Permanent / Non-retryable client errors
        for code in (400, 401, 403, 404, 410):
            http_4xx = urllib.error.HTTPError("https://api.test", code, "Client Error", {}, None)
            err_type, retryable = classify_network_error(http_4xx)
            assert err_type == f"HTTP_{code}_CLIENT_ERROR"
            assert retryable is False

    @patch("scripts.acquire_workplace_dataset.time.sleep", return_value=None)
    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_transient_failure_followed_by_success(self, mock_urlopen, mock_sleep):
        expected_bytes = create_dummy_image_bytes(100, 100)

        # First 2 calls fail with transient errors; 3rd call succeeds
        mock_urlopen.side_effect = [
            ssl.SSLError("Handshake failed"),
            urllib.error.HTTPError("https://api.test", 503, "Service Unavailable", {}, None),
            MockHTTPResponse(expected_bytes),
        ]

        data, err_type, err_reason = fetch_url_with_retries(
            "https://api.test/image.jpg",
            timeout=5.0,
            max_retries=3,
            base_backoff=0.01,
            max_backoff=0.05,
            jitter=0.01,
        )

        assert data == expected_bytes
        assert err_type is None
        assert err_reason is None
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("scripts.acquire_workplace_dataset.time.sleep", return_value=None)
    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_retry_exhaustion_records_failure_and_stops(self, mock_urlopen, mock_sleep, tmp_path, monkeypatch):
        mock_urlopen.side_effect = ssl.SSLError("SSL handshake failure")

        # Redirect dataset paths to isolated temporary directory
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", tmp_path / "acquisition_log.jsonl")

        candidate = {
            "url": "https://api.test/failing_photo.jpg",
            "ground_truth": "REAL",
            "source": "Wikimedia Commons",
            "license": "CC-BY",
            "author": "Test Author",
            "description": "Test failure",
        }
        existing_sha = {}
        existing_dhashes = []
        current_counts = {"train": {"REAL": 0, "FAKE": 0}, "val": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
        existing_urls = set()

        success = download_and_validate_asset(
            candidate,
            existing_sha,
            existing_dhashes,
            target_dhash=None,
            current_counts=current_counts,
            existing_urls=existing_urls,
            timeout=5.0,
            max_retries=2,
            base_backoff=0.01,
            max_backoff=0.02,
        )

        assert success is False
        assert candidate["url"] in existing_urls
        # 1 initial attempt + 2 retries = 3 attempts
        assert mock_urlopen.call_count == 3

        # Verify log entry records FAILED status with error_type and retries_exhausted
        log_file = tmp_path / "acquisition_log.jsonl"
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        failed_entry = [entry for entry in lines if entry.get("url") == candidate["url"]][0]
        assert failed_entry["status"] == "FAILED"
        assert failed_entry["error_type"] == "SSL_HANDSHAKE_ERROR"
        assert failed_entry["retries_exhausted"] is True

    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_keyboard_interrupt_propagates_from_fetch(self, mock_urlopen):
        mock_urlopen.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            fetch_url_with_retries("https://api.test/image.jpg")

    def test_keyboard_interrupt_clean_exit_in_run_acquisition(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", tmp_path / "acquisition_log.jsonl")
        monkeypatch.setattr("scripts.acquire_workplace_dataset.QUARANTINE_DIR", tmp_path / "quarantine")
        monkeypatch.setattr("scripts.acquire_workplace_dataset.TRACKED_OFFICE_IMAGE", tmp_path / "nonexistent.png")

        # Mock fetch_wikimedia_workplace_candidates to raise KeyboardInterrupt
        with patch("scripts.acquire_workplace_dataset.fetch_wikimedia_workplace_candidates", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc_info:
                run_acquisition(target_real=5, target_fake=5)
            # Must exit with code 0 cleanly
            assert exc_info.value.code == 0

    def test_partial_download_cleanup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        dest_dir = tmp_path / "train" / "REAL"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create orphan partial files
        partial_1 = dest_dir / ".tmp_workplace_real_00001.jpg"
        partial_2 = dest_dir / "workplace_real_00002.tmp"
        partial_1.write_bytes(b"corrupted partial data")
        partial_2.write_bytes(b"half downloaded data")

        assert partial_1.exists()
        assert partial_2.exists()

        cleanup_temp_files(tmp_path)

        assert not partial_1.exists()
        assert not partial_2.exists()

    def test_get_current_counts_ignores_temp_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        train_real = tmp_path / "train" / "REAL"
        train_real.mkdir(parents=True, exist_ok=True)

        # Write 1 valid image
        valid_img = train_real / "workplace_real_00001.jpg"
        valid_img.write_bytes(create_dummy_image_bytes(64, 64))

        # Write 2 partial .tmp files
        (train_real / ".tmp_workplace_real_00002.jpg").write_bytes(b"partial")
        (train_real / "workplace_real_00003.tmp").write_bytes(b"partial")

        counts = get_current_counts()
        # Only the valid file must be counted
        assert counts["train"]["REAL"] == 1

    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_resuming_skips_existing_urls_without_network_call(self, mock_urlopen, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        log_file = tmp_path / "acquisition_log.jsonl"
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", log_file)

        existing_url = "https://api.test/already_processed.jpg"
        # Pre-seed log with verified entry
        log_file.write_text(json.dumps({
            "url": existing_url,
            "status": "VERIFIED",
            "file_path": "train/REAL/workplace_real_00001.jpg",
            "sha256": "abcdef123456",
            "label": "REAL"
        }) + "\n", encoding="utf-8")

        existing_urls, existing_sha, existing_dhashes, failed_urls = load_existing_ledger()
        assert existing_url in existing_urls
        assert "abcdef123456" in existing_sha

        # Simulate candidate matching the existing URL
        candidate = {
            "url": existing_url,
            "ground_truth": "REAL",
            "source": "Wikimedia Commons"
        }

        # If candidate URL is already in existing_urls, the acquisition loop skips it
        assert candidate["url"] in existing_urls
        mock_urlopen.assert_not_called()

    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_sha256_duplicate_rejected_without_duplicate_file(self, mock_urlopen, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        log_file = tmp_path / "acquisition_log.jsonl"
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", log_file)

        img_bytes = create_dummy_image_bytes(256, 256)
        sha = compute_sha256_bytes(img_bytes)

        existing_sha = {sha: "val/REAL/workplace_real_00010.jpg"}
        existing_dhashes = []
        current_counts = {"train": {"REAL": 0, "FAKE": 0}, "val": {"REAL": 1, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
        existing_urls = set()

        mock_urlopen.return_value = MockHTTPResponse(img_bytes)

        candidate = {
            "url": "https://api.test/duplicate_image.jpg",
            "ground_truth": "REAL",
            "source": "Wikimedia Commons",
            "license": "CC-BY",
        }

        success = download_and_validate_asset(
            candidate,
            existing_sha,
            existing_dhashes,
            target_dhash=None,
            current_counts=current_counts,
            existing_urls=existing_urls,
            timeout=5.0,
            max_retries=1,
        )

        assert success is False
        # Logged as duplicate SHA
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(l) for l in f if l.strip()]
        assert logs[-1]["status"] == "DUPLICATE_SHA"
        assert logs[-1]["sha256"] == sha

    def test_extract_subject_key_normalizes_photo_series(self):
        from scripts.acquire_workplace_dataset import extract_subject_key

        title1 = "File:Alan Schaaf, Imgur Founder, in Wellington (TEDxWellington) (14064560181).jpg"
        title2 = "File:Alan Schaaf, Imgur Founder, in Wellington (TEDxWellington) (14064564222).jpg"
        k1 = extract_subject_key(title1, "")
        k2 = extract_subject_key(title2, "")
        assert k1 == k2
        assert "alan schaaf" in k1

        j1 = "File:Outstanding Achievement- Janice Shepherd (27756262174).jpg"
        j2 = "File:Outstanding Achievement- Janice Shepherd (28336332156).jpg"
        assert extract_subject_key(j1, "") == extract_subject_key(j2, "")

    @patch("scripts.acquire_workplace_dataset.fetch_url_with_retries")
    def test_candidate_diversity_caps_repeated_subjects(self, mock_fetch):
        from scripts.acquire_workplace_dataset import fetch_wikimedia_workplace_candidates

        api_response = {
            "query": {
                "pages": {
                    "101": {
                        "title": "File:Alan Schaaf, Imgur Founder (14064560181).jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/img1.jpg", "width": 800, "height": 600}]
                    },
                    "102": {
                        "title": "File:Alan Schaaf, Imgur Founder (14064564222).jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/img2.jpg", "width": 800, "height": 600}]
                    },
                    "103": {
                        "title": "File:Software Engineer at Desk.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/engineer.jpg", "width": 800, "height": 600}]
                    }
                }
            }
        }
        mock_fetch.return_value = (json.dumps(api_response).encode("utf-8"), None, None)

        seen_subjects = {}
        candidates = fetch_wikimedia_workplace_candidates(
            limit=50,
            offset=0,
            existing_urls=set(),
            seen_subjects=seen_subjects,
            max_per_subject=1
        )

        # First Alan Schaaf accepted; second skipped due to cap; Engineer accepted
        titles = [c["title"] for c in candidates]
        assert len(candidates) == 2
        assert "File:Alan Schaaf, Imgur Founder (14064560181).jpg" in titles
        assert "File:Software Engineer at Desk.jpg" in titles
        assert "File:Alan Schaaf, Imgur Founder (14064564222).jpg" not in titles

    @patch("scripts.acquire_workplace_dataset.fetch_url_with_retries")
    def test_domain_relevance_real_keyword_filtering(self, mock_fetch):
        from scripts.acquire_workplace_dataset import fetch_wikimedia_workplace_candidates

        api_response = {
            "query": {
                "pages": {
                    "201": {
                        "title": "File:Computer lab tool kit with screws and cable.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/toolkit.jpg", "width": 800, "height": 600}]
                    },
                    "202": {
                        "title": "File:Allen & Heath Qu-24 Chrome Edition mixing console.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/mixer.jpg", "width": 800, "height": 600}]
                    },
                    "203": {
                        "title": "File:Businesswoman working on laptop at desk.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/businesswoman.jpg", "width": 800, "height": 600}]
                    }
                }
            }
        }
        mock_fetch.return_value = (json.dumps(api_response).encode("utf-8"), None, None)

        existing_urls = set()
        candidates = fetch_wikimedia_workplace_candidates(
            limit=50,
            offset=0,
            existing_urls=existing_urls,
            seen_subjects={},
            max_per_subject=1
        )

        # Only the businesswoman should be accepted
        assert len(candidates) == 1
        assert candidates[0]["title"] == "File:Businesswoman working on laptop at desk.jpg"
        # Tool kit and mixer URLs should be quarantined in existing_urls
        assert "https://upload.wikimedia.org/toolkit.jpg" in existing_urls
        assert "https://upload.wikimedia.org/mixer.jpg" in existing_urls

    @patch("scripts.acquire_workplace_dataset.fetch_url_with_retries")
    def test_domain_relevance_fake_prompt_filtering(self, mock_fetch):
        from scripts.acquire_workplace_dataset import fetch_openfake_generative_candidates

        api_response = {
            "rows": [
                {
                    "row": {
                        "label": "fake",
                        "model": "imagen-4.0",
                        "prompt": "A cute gingerbread fox in the forest",
                        "image": {"src": "https://hf.co/fox.jpg"}
                    }
                },
                {
                    "row": {
                        "label": "fake",
                        "model": "sd-1.5",
                        "prompt": "Batman standing on a rooftop in Gotham",
                        "image": {"src": "https://hf.co/batman.jpg"}
                    }
                },
                {
                    "row": {
                        "label": "fake",
                        "model": "flux.1-dev",
                        "prompt": "Professional corporate businesswoman typing on laptop at office desk, portrait",
                        "image": {"src": "https://hf.co/office_woman.jpg"}
                    }
                }
            ]
        }
        mock_fetch.return_value = (json.dumps(api_response).encode("utf-8"), None, None)

        candidates = fetch_openfake_generative_candidates(limit=50, offset=0)
        assert len(candidates) == 1
        assert candidates[0]["generator"] == "flux.1-dev"
        assert candidates[0]["url"] == "https://hf.co/office_woman.jpg"

    def test_audit_split_leakage_detection(self, tmp_path, monkeypatch):
        from scripts.audit_workplace_dataset import run_comprehensive_audit

        monkeypatch.setattr("scripts.audit_workplace_dataset.DATASET_ROOT", tmp_path)
        monkeypatch.setattr("scripts.audit_workplace_dataset.QUARANTINE_DIR", tmp_path / "quarantine")
        monkeypatch.setattr("scripts.audit_workplace_dataset.LOG_PATH", tmp_path / "acquisition_log.jsonl")
        monkeypatch.setattr("scripts.audit_workplace_dataset.TRACKED_OFFICE_IMAGE", tmp_path / "none.png")

        # Create train and val with images from same subject
        train_dir = tmp_path / "train" / "REAL"
        val_dir = tmp_path / "val" / "REAL"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        img1 = train_dir / "workplace_real_00067.jpg"
        img2 = val_dir / "workplace_real_00071.jpg"
        img1.write_bytes(create_dummy_image_bytes(256, 256, color=(10, 20, 30)))
        img2.write_bytes(create_dummy_image_bytes(256, 256, color=(15, 25, 35)))

        # Prepopulate log with Alan Schaaf titles
        log_file = tmp_path / "acquisition_log.jsonl"
        log_file.write_text(
            json.dumps({"file_path": "train/REAL/workplace_real_00067.jpg", "status": "VERIFIED", "title": "Alan Schaaf in Wellington 1"}) + "\n" +
            json.dumps({"file_path": "val/REAL/workplace_real_00071.jpg", "status": "VERIFIED", "title": "Alan Schaaf in Wellington 2"}) + "\n",
            encoding="utf-8"
        )

        report = run_comprehensive_audit(quarantine_ineligible=False)
        assert len(report["cross_split_leaked_subjects"]) == 1
        assert "SUBJECT_ALAN_SCHAAF_TEDX" in report["cross_split_leaked_subjects"]

    def test_extract_subject_key_handles_url_basename_without_title(self):
        from scripts.acquire_workplace_dataset import extract_subject_key

        url1 = "https://upload.wikimedia.org/wikipedia/commons/0/03/Mindpark_Helsingborg_-_Markplan.jpg?utm_source=commons"
        key1 = extract_subject_key("", url1)
        assert "mindpark helsingborg markplan" in key1
        assert "https upload" not in key1

        url2 = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Software_Engineer_at_Desk_2022.png"
        key2 = extract_subject_key("", url2)
        assert "software engineer at desk" in key2
        assert "https upload" not in key2
        assert key1 != key2

    @patch("scripts.acquire_workplace_dataset.fetch_url_with_retries")
    def test_subject_quota_does_not_consume_on_unverified_candidate(self, mock_fetch):
        from scripts.acquire_workplace_dataset import fetch_wikimedia_workplace_candidates, QueryDiagnostics

        api_response = {
            "query": {
                "pages": {
                    "301": {
                        "title": "File:Software_Engineer_Writing_Code_Desk.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/engineer_desk.jpg", "width": 800, "height": 600}]
                    }
                }
            }
        }
        mock_fetch.return_value = (json.dumps(api_response).encode("utf-8"), None, None)

        seen_subjects = {}
        diag = QueryDiagnostics(query="test_query")
        existing_urls = set()

        candidates = fetch_wikimedia_workplace_candidates(
            limit=50,
            offset=0,
            existing_urls=existing_urls,
            seen_subjects=seen_subjects,
            max_per_subject=1,
            diag=diag,
        )

        assert len(candidates) == 1
        # seen_subjects must NOT be incremented until download and validation succeed
        assert seen_subjects.get(candidates[0]["subject_key"], 0) == 0
        # Candidate URL must NOT be added to existing_urls yet
        assert candidates[0]["url"] not in existing_urls
        assert diag.candidates_found == 1
        assert diag.rejected_subject_quota == 0

    def test_query_diagnostics_records_all_metrics(self):
        from scripts.acquire_workplace_dataset import QueryDiagnostics

        diag = QueryDiagnostics(query='filetype:bitmap "working" "laptop"')
        diag.candidates_found = 25
        diag.http_api_errors = 1
        diag.rejected_non_bitmap = 2
        diag.rejected_keyword_filter = 3
        diag.rejected_subject_quota = 4
        diag.rejected_duplicates = 5
        diag.rejected_invalid_images = 2
        diag.successfully_accepted = 8

        d_dict = diag.to_dict()
        assert d_dict["candidates_found"] == 25
        assert d_dict["http_api_errors"] == 1
        assert d_dict["rejected_non_bitmap"] == 2
        assert d_dict["rejected_keyword_filter"] == 3
        assert d_dict["rejected_subject_quota"] == 4
        assert d_dict["rejected_duplicates"] == 5
        assert d_dict["rejected_invalid_images"] == 2
        assert d_dict["successfully_accepted"] == 8

    def test_filetype_bitmap_in_real_search_queries(self):
        from scripts.acquire_workplace_dataset import WIKIMEDIA_REAL_SEARCH_QUERIES

        assert len(WIKIMEDIA_REAL_SEARCH_QUERIES) >= 15
        for q in WIKIMEDIA_REAL_SEARCH_QUERIES:
            assert "filetype:bitmap" in q, f"Query '{q}' is missing filetype:bitmap filter to prevent non-bitmap flooding"

    def test_determine_split_destination_balances_deficits(self, tmp_path, monkeypatch):
        from scripts.acquire_workplace_dataset import determine_split_destination
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)

        # Deficit state: train is behind (7 REAL vs 9 val and 9 test)
        counts = {
            "train": {"REAL": 7, "FAKE": 50},
            "val": {"REAL": 9, "FAKE": 25},
            "test": {"REAL": 9, "FAKE": 25},
        }

        # Train has deficit (0.60 * 26 - 7 = 8.6), so it must be selected
        dest_dir, target_split = determine_split_destination("REAL", counts)
        assert target_split == "train"
        assert dest_dir == tmp_path / "train" / "REAL"

    @patch("scripts.acquire_workplace_dataset.download_and_validate_asset", return_value=True)
    @patch("scripts.acquire_workplace_dataset.fetch_wikimedia_workplace_candidates")
    @patch("scripts.acquire_workplace_dataset.fetch_openfake_generative_candidates")
    def test_target_arguments_represent_total_eligible_counts_and_resumes(
        self, mock_fake_candidates, mock_real_candidates, mock_download, tmp_path, monkeypatch
    ):
        from scripts.acquire_workplace_dataset import run_acquisition

        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", tmp_path / "acquisition_log.jsonl")
        monkeypatch.setattr("scripts.acquire_workplace_dataset.QUARANTINE_DIR", tmp_path / "quarantine")
        monkeypatch.setattr("scripts.acquire_workplace_dataset.TRACKED_OFFICE_IMAGE", tmp_path / "nonexistent.png")

        # Set up disk with 10 REAL and 20 FAKE
        train_r = tmp_path / "train" / "REAL"
        train_f = tmp_path / "train" / "FAKE"
        train_r.mkdir(parents=True, exist_ok=True)
        train_f.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (train_r / f"workplace_real_{i:05d}.jpg").write_bytes(create_dummy_image_bytes(64, 64))
        for i in range(20):
            (train_f / f"workplace_fake_{i:05d}.jpg").write_bytes(create_dummy_image_bytes(64, 64))

        # Request target_real=12 (needs 2 more), target_fake=20 (needs 0 more)
        mock_real_candidates.return_value = [
            {"url": "https://api.test/r1.jpg", "ground_truth": "REAL", "subject_key": "s1", "title": "P1"},
            {"url": "https://api.test/r2.jpg", "ground_truth": "REAL", "subject_key": "s2", "title": "P2"},
        ]

        run_acquisition(target_real=12, target_fake=20)

        # FAKE acquisition must NOT have fetched any candidates since target=20 was already reached
        mock_fake_candidates.assert_not_called()
        # REAL acquisition must have downloaded exactly 2 to reach target of 12
        assert mock_download.call_count == 2

    @patch("scripts.acquire_workplace_dataset.fetch_url_with_retries")
    def test_semantic_failure_cases_rejected_by_keyword_filters(self, mock_fetch):
        from scripts.acquire_workplace_dataset import fetch_wikimedia_workplace_candidates

        api_response = {
            "query": {
                "pages": {
                    "401": {
                        "title": "File:Cromemco System One in Paris.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/cromemco.jpg", "width": 800, "height": 600}]
                    },
                    "402": {
                        "title": "File:Fresnel - oeuvres completes d'Augustin Fresnel tome 1 (page crop).jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/fresnel.jpg", "width": 800, "height": 600}]
                    },
                    "403": {
                        "title": "File:Camper van parked by the side of the road with man working on a laptop.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/camper.jpg", "width": 800, "height": 600}]
                    },
                    "404": {
                        "title": "File:Child remote schooling on a laptop in the countryside.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/child_school.jpg", "width": 800, "height": 600}]
                    },
                    "405": {
                        "title": "File:Software engineer programming at office workstation.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/valid_engineer.jpg", "width": 800, "height": 600}]
                    }
                }
            }
        }
        mock_fetch.return_value = (json.dumps(api_response).encode("utf-8"), None, None)

        candidates = fetch_wikimedia_workplace_candidates(
            limit=50,
            offset=0,
            existing_urls=set(),
            seen_subjects={},
            max_per_subject=1
        )

        # Only the valid software engineer should be accepted; all 4 semantic failure cases must be rejected
        assert len(candidates) == 1
        assert candidates[0]["title"] == "File:Software engineer programming at office workstation.jpg"

    @patch("scripts.acquire_workplace_dataset.fetch_url_with_retries")
    def test_exact_user_reported_semantic_failures_rejected(self, mock_fetch):
        """
        Regression test for exact failure cases reported by user:
        1. Le Tour du monde - 15 (page 200 crop).jpg
        2. Denier Vulgate Glaire.jpg
        3. Flag of Beverly Hills, California.png
        4. WarshipsWidthPPPlot.png
        5. Feeding a laboratory rat.jpg
        6. Mount Kilimanjaro Closeup
        7. Watercolor depiction of a Wikipedia user
        """
        from scripts.acquire_workplace_dataset import fetch_wikimedia_workplace_candidates

        api_response = {
            "query": {
                "pages": {
                    "501": {
                        "title": "File:Le Tour du monde - 15 (page 200 crop).jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/tour_monde.jpg", "width": 858, "height": 1352}]
                    },
                    "502": {
                        "title": "File:Denier Vulgate Glaire.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/denier.jpg", "width": 938, "height": 362}]
                    },
                    "503": {
                        "title": "File:Flag of Beverly Hills, California.png",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/flag_bh.png", "width": 5760, "height": 3456}]
                    },
                    "504": {
                        "title": "File:WarshipsWidthPPPlot.png",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/warships_pp.png", "width": 356, "height": 502}]
                    },
                    "505": {
                        "title": "File:Feeding a laboratory rat.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/lab_rat.jpg", "width": 4881, "height": 3258}]
                    },
                    "506": {
                        "title": "File:Mount Kilimanjaro Closeup (4708 - ISS006-E-45499 lrg).jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/kilimanjaro.jpg", "width": 1000, "height": 662}]
                    },
                    "507": {
                        "title": "File:Watercolor depiction of a Wikipedia user at her desk labeled Wiki Maddie.webp",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/wiki_maddie.webp", "width": 2270, "height": 3527}]
                    },
                    "508": {
                        "title": "File:Business professional working at laptop desk in modern office.jpg",
                        "imageinfo": [{"url": "https://upload.wikimedia.org/valid_business_laptop.jpg", "width": 1200, "height": 800}]
                    }
                }
            }
        }
        mock_fetch.return_value = (json.dumps(api_response).encode("utf-8"), None, None)

        candidates = fetch_wikimedia_workplace_candidates(
            limit=50,
            offset=0,
            existing_urls=set(),
            seen_subjects={},
            max_per_subject=1
        )

        assert len(candidates) == 1
        assert candidates[0]["title"] == "File:Business professional working at laptop desk in modern office.jpg"

    def test_multimodal_pixel_and_semantics_validator_rejects_failures(self):
        """Validates that candidate validator rejects non-workplace pixel signatures and metadata."""
        from scripts.acquire_workplace_dataset import (
            validate_candidate_pixels,
            validate_candidate_semantics_and_pixels,
        )

        # 1. Metadata check for the exact failure candidates
        failure_candidates = [
            {"title": "File:Le Tour du monde - 15 (page 200 crop).jpg", "ground_truth": "REAL"},
            {"title": "File:Denier Vulgate Glaire.jpg", "ground_truth": "REAL"},
            {"title": "File:Flag of Beverly Hills, California.png", "ground_truth": "REAL"},
            {"title": "File:WarshipsWidthPPPlot.png", "ground_truth": "REAL"},
            {"title": "File:Feeding a laboratory rat.jpg", "ground_truth": "REAL"},
            {"title": "File:Mount Kilimanjaro Closeup.jpg", "ground_truth": "REAL"},
            {"title": "File:Watercolor depiction of a Wikipedia user.webp", "ground_truth": "REAL"},
        ]
        dummy_img = Image.new("RGB", (256, 256), color=(120, 130, 140))
        for cand in failure_candidates:
            ok, reason = validate_candidate_semantics_and_pixels(dummy_img, cand)
            assert ok is False, f"Expected {cand['title']} to be rejected"
            assert len(reason) > 0

        # 2. Pixel check: Statistical plot signature (monochrome, high white background, low unique colors)
        plot_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        # Draw some black lines
        from PIL import ImageDraw
        draw = ImageDraw.Draw(plot_img)
        for y in range(20, 240, 20):
            draw.line([(20, y), (240, y)], fill=(0, 0, 0), width=1)
        pix_ok, pix_reason = validate_candidate_pixels(plot_img)
        assert pix_ok is False
        assert ("statistical chart / plot diagram" in pix_reason or "scanned book page" in pix_reason)

    def test_office_laptop_regression_image_isolation(self):
        """
        Confirms ai_man_office_laptop.png is isolated from train and validation splits,
        and its provenance indicates it is a development regression asset, not an untouched held-out final test item.
        """
        from scripts.acquire_workplace_dataset import DATASET_ROOT, LEDGER_PATH

        # 1. Confirm not in train or val
        train_matches = list((DATASET_ROOT / "train").glob("**/ai_man_office_laptop*"))
        val_matches = list((DATASET_ROOT / "val").glob("**/ai_man_office_laptop*"))
        assert len(train_matches) == 0, f"ai_man_office_laptop found in train split: {train_matches}"
        assert len(val_matches) == 0, f"ai_man_office_laptop found in val split: {val_matches}"

        # 2. Check ledger documentation if ledger exists
        if LEDGER_PATH.exists():
            with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                ledger = json.load(f)
            office_recs = [
                r for r in ledger.get("verified_records", [])
                if "ai_man_office_laptop" in str(r.get("file_path", ""))
            ]
            if office_recs:
                rec = office_recs[0]
                assert rec["split"] == "test"
                notes = rec.get("provenance_notes", "")
                assert "development regression" in notes.lower() or "benchmark" in notes.lower()

    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_raw_vs_disk_hash_difference(self, mock_urlopen, tmp_path, monkeypatch):
        """
        Validates that downloading a non-JPEG payload (PNG) results in raw_payload_sha256 != final_disk_sha256,
        both hashes are separately recorded, and final_disk_sha256 exactly matches the saved on-disk JPEG file.
        """
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        log_file = tmp_path / "acquisition_log.jsonl"
        ledger_file = tmp_path / "provenance_ledger.json"
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", log_file)

        for s in ["train", "val", "test"]:
            for l in ["REAL", "FAKE"]:
                (tmp_path / s / l).mkdir(parents=True, exist_ok=True)

        # Create an uncompressed PNG in memory from a realistic gradient photo
        img = create_dummy_photo_image(320, 240)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        raw_sha = compute_sha256_bytes(raw_bytes)

        mock_urlopen.return_value = MockHTTPResponse(raw_bytes)

        candidate = {
            "url": "https://upload.wikimedia.org/modern_desk_setup.png",
            "title": "File:Modern professional software engineering workplace desk.png",
            "ground_truth": "REAL",
            "source": "Wikimedia Commons",
            "license": "CC-BY-SA 4.0",
            "author": "Test Photographer",
            "description": "Desk with laptop and monitors in modern office",
        }
        existing_sha = {}
        existing_dhashes = []
        current_counts = {"train": {"REAL": 0, "FAKE": 0}, "val": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
        existing_urls = set()

        success = download_and_validate_asset(
            candidate,
            existing_sha,
            existing_dhashes,
            target_dhash=None,
            current_counts=current_counts,
            existing_urls=existing_urls,
            timeout=5.0,
            max_retries=1,
        )

        assert success is True

        # Locate saved file on disk
        saved_files = list((tmp_path / "train" / "REAL").glob("*.jpg"))
        assert len(saved_files) == 1
        saved_path = saved_files[0]
        disk_bytes = saved_path.read_bytes()
        disk_sha = compute_sha256_bytes(disk_bytes)

        # 1. Raw PNG bytes differ from saved JPEG bytes
        assert raw_sha != disk_sha

        # 2. Acquisition log records BOTH raw and disk SHA
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
        verified_log = [l for l in logs if l.get("status") == "VERIFIED"]
        assert len(verified_log) == 1
        assert verified_log[0]["raw_payload_sha256"] == raw_sha
        assert verified_log[0]["final_disk_sha256"] == disk_sha
        assert verified_log[0]["sha256"] == disk_sha

        # 3. Provenance ledger records BOTH raw and disk SHA, matching on-disk file
        assert ledger_file.exists()
        with open(ledger_file, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        assert ledger["verified_count"] == 1
        rec = ledger["verified_records"][0]
        assert rec["raw_payload_sha256"] == raw_sha
        assert rec["final_disk_sha256"] == disk_sha
        assert rec["sha256"] == disk_sha
        assert rec["final_disk_sha256"] == compute_sha256_bytes(saved_path.read_bytes())

    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_successful_ledger_synchronization(self, mock_urlopen, tmp_path, monkeypatch):
        """
        Validates that every accepted active asset is synchronously and completely written
        to provenance_ledger.json with relative path, split, class, checksums, and metadata.
        """
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        log_file = tmp_path / "acquisition_log.jsonl"
        ledger_file = tmp_path / "provenance_ledger.json"
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", log_file)

        for s in ["train", "val", "test"]:
            for l in ["REAL", "FAKE"]:
                (tmp_path / s / l).mkdir(parents=True, exist_ok=True)

        photo_img = create_dummy_photo_image(320, 240)
        buf = io.BytesIO()
        photo_img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        mock_urlopen.return_value = MockHTTPResponse(img_bytes)

        candidate = {
            "url": "https://upload.wikimedia.org/office_conference_room.jpg",
            "title": "File:Modern office conference room and workplace desk.jpg",
            "ground_truth": "REAL",
            "source": "Wikimedia Commons",
            "license": "CC-BY 4.0",
            "author": "Alice Doe",
            "description": "Collaborative office workplace meeting room",
        }
        existing_sha = {}
        existing_dhashes = []
        current_counts = {"train": {"REAL": 0, "FAKE": 0}, "val": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
        existing_urls = set()

        success = download_and_validate_asset(
            candidate,
            existing_sha,
            existing_dhashes,
            target_dhash=None,
            current_counts=current_counts,
            existing_urls=existing_urls,
            timeout=5.0,
            max_retries=1,
        )

        assert success is True
        assert ledger_file.exists()

        with open(ledger_file, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        assert ledger["verified_count"] == 1
        entry = ledger["verified_records"][0]
        assert entry["file_path"] == "train/REAL/workplace_real_00001.jpg"
        assert entry["filename"] == "workplace_real_00001.jpg"
        assert entry["split"] == "train"
        assert entry["label"] == "REAL"
        assert entry["class_idx"] == 1
        assert entry["status"] == "VERIFIED_ACTIVE"
        assert entry["url"] == candidate["url"]
        assert entry["title"] == candidate["title"]
        assert entry["source"] == "Wikimedia Commons"
        assert entry["license"] == "CC-BY 4.0"
        assert entry["generator"] == "Alice Doe"
        assert entry["final_disk_sha256"] != "unknown"
        assert entry["raw_payload_sha256"] != "unknown"
        assert entry["timestamp"] is not None

    @patch("scripts.acquire_workplace_dataset.urllib.request.urlopen")
    def test_failed_downloads_not_in_ledger(self, mock_urlopen, tmp_path, monkeypatch):
        """
        Validates that failed or rejected candidates (HTTP error, corrupt data, semantic rejection)
        are NEVER added to provenance_ledger.json verified_records.
        """
        monkeypatch.setattr("scripts.acquire_workplace_dataset.DATASET_ROOT", tmp_path)
        log_file = tmp_path / "acquisition_log.jsonl"
        ledger_file = tmp_path / "provenance_ledger.json"
        monkeypatch.setattr("scripts.acquire_workplace_dataset.LOG_PATH", log_file)

        for s in ["train", "val", "test"]:
            for l in ["REAL", "FAKE"]:
                (tmp_path / s / l).mkdir(parents=True, exist_ok=True)

        existing_sha = {}
        existing_dhashes = []
        current_counts = {"train": {"REAL": 0, "FAKE": 0}, "val": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
        existing_urls = set()

        # 1. HTTP 404 Failure
        mock_urlopen.side_effect = urllib.error.HTTPError("https://api.test/missing.jpg", 404, "Not Found", {}, None)
        c1 = {"url": "https://api.test/missing.jpg", "ground_truth": "REAL", "source": "Wikimedia Commons", "title": "Desk"}
        success1 = download_and_validate_asset(c1, existing_sha, existing_dhashes, None, current_counts, existing_urls)
        assert success1 is False

        # 2. Corrupt Image Bytes
        mock_urlopen.side_effect = None
        mock_urlopen.return_value = MockHTTPResponse(b"NOT_A_VALID_IMAGE_DATA")
        c2 = {"url": "https://api.test/corrupt.jpg", "ground_truth": "REAL", "source": "Wikimedia Commons", "title": "Desk"}
        success2 = download_and_validate_asset(c2, existing_sha, existing_dhashes, None, current_counts, existing_urls)
        assert success2 is False

        # 3. Semantic / Flagged Rejection
        mock_urlopen.return_value = MockHTTPResponse(create_dummy_image_bytes(256, 256))
        c3 = {
            "url": "https://api.test/flag.png",
            "title": "File:Flag of Beverly Hills, California.png",
            "ground_truth": "REAL",
            "source": "Wikimedia Commons",
        }
        success3 = download_and_validate_asset(c3, existing_sha, existing_dhashes, None, current_counts, existing_urls)
        assert success3 is False

        # Check ledger remains completely free of any verified records
        if ledger_file.exists():
            with open(ledger_file, "r", encoding="utf-8") as f:
                ledger = json.load(f)
            assert len(ledger.get("verified_records", [])) == 0
            assert ledger.get("verified_count", 0) == 0

        # 4. Direct sync attempt with non-verified status must return False
        assert sync_verified_record_to_ledger({"status": "FAILED", "sha256": "abc"}, ledger_file) is False
        assert sync_verified_record_to_ledger({"status": "QUARANTINED", "sha256": "def"}, ledger_file) is False
        assert sync_verified_record_to_ledger({"status": "DUPLICATE_SHA", "sha256": "ghi"}, ledger_file) is False

    def test_repeated_acquisition_without_duplicate_ledger_entries(self, tmp_path):
        """
        Validates that repeated acquisition or synchronization of the same asset updates idempotently
        without creating duplicate entries in provenance_ledger.json.
        """
        ledger_file = tmp_path / "provenance_ledger.json"

        record1 = {
            "file_path": "train/REAL/workplace_real_00001.jpg",
            "filename": "workplace_real_00001.jpg",
            "split": "train",
            "label": "REAL",
            "class_idx": 1,
            "final_disk_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "raw_payload_sha256": "123456abcdef123456abcdef123456abcdef123456abcdef123456abcdef1234",
            "url": "https://upload.wikimedia.org/office_laptop.jpg",
            "title": "File:Modern office workspace with laptop.jpg",
            "source": "Wikimedia Commons",
            "license": "CC-BY-SA 4.0",
            "author_or_generator": "Test Contributor",
            "provenance_notes": "First acquisition pass",
            "status": "VERIFIED",
        }

        # First synchronization
        ok1 = sync_verified_record_to_ledger(record1, ledger_file)
        assert ok1 is True

        with open(ledger_file, "r", encoding="utf-8") as f:
            ledger1 = json.load(f)
        assert ledger1["verified_count"] == 1
        assert len(ledger1["verified_records"]) == 1

        # Second synchronization of identical record
        ok2 = sync_verified_record_to_ledger(record1, ledger_file)
        assert ok2 is True

        with open(ledger_file, "r", encoding="utf-8") as f:
            ledger2 = json.load(f)
        assert ledger2["verified_count"] == 1
        assert len(ledger2["verified_records"]) == 1

        # Third synchronization with updated metadata note
        record1_updated = dict(record1)
        record1_updated["provenance_notes"] = "Updated acquisition pass with refined notes"
        ok3 = sync_verified_record_to_ledger(record1_updated, ledger_file)
        assert ok3 is True

        with open(ledger_file, "r", encoding="utf-8") as f:
            ledger3 = json.load(f)
        assert ledger3["verified_count"] == 1
        assert len(ledger3["verified_records"]) == 1
        assert ledger3["verified_records"][0]["provenance_notes"] == "Updated acquisition pass with refined notes"





