"""
Regression Tests for Provenance Ledger Backfill & Reconciliation
================================================================
Validates:
1. Duplicate ledger records detection and prevention.
2. Cross-platform path normalization (forward slashes, matching split/class hierarchy).
3. Hash accuracy and separation: final_disk_sha256 matches disk bytes; raw_payload_sha256 kept separate.
4. Missing provenance explicitly marked 'unknown' without fabrication.
5. Conflicting hashes or labels detected and reported rather than overwritten.
6. Repeated execution is completely idempotent and produces identical record counts.
7. Quarantine records and count (156) remain intact.
"""

import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from PIL import Image

import pytest

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from scripts.backfill_provenance_ledger import run_backfill, compute_sha256_file


def create_dummy_image(width=100, height=100, color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def temp_dataset_env(tmp_path):
    """Creates an isolated mock workplace_dataset directory with active files and logs."""
    dataset_dir = tmp_path / "workplace_dataset"
    for s in ["train", "val", "test"]:
        for c in ["REAL", "FAKE"]:
            (dataset_dir / s / c).mkdir(parents=True, exist_ok=True)

    # Create 3 active files
    file1 = dataset_dir / "train" / "REAL" / "workplace_real_00001.jpg"
    b1 = create_dummy_image(120, 120, (10, 20, 30))
    file1.write_bytes(b1)

    file2 = dataset_dir / "val" / "FAKE" / "workplace_fake_00001.jpg"
    b2 = create_dummy_image(140, 140, (40, 50, 60))
    file2.write_bytes(b2)

    file3 = dataset_dir / "test" / "REAL" / "local_real_legacy.jpg"
    b3 = create_dummy_image(160, 160, (70, 80, 90))
    file3.write_bytes(b3)

    # Create acquisition log with entries for file1 and file2
    log_file = dataset_dir / "acquisition_log.jsonl"
    log_entries = [
        {
            "status": "VERIFIED",
            "file_path": "train/REAL/workplace_real_00001.jpg",
            "filename": "workplace_real_00001.jpg",
            "split": "train",
            "label": "REAL",
            "class_idx": 1,
            "sha256": "raw_payload_hash_111111",
            "raw_payload_sha256": "raw_payload_hash_111111",
            "url": "https://upload.wikimedia.org/office1.jpg",
            "title": "Office Desk",
            "source": "Wikimedia Commons",
            "license": "CC-BY-SA 4.0",
            "author_or_generator": "Photographer One",
            "timestamp": "2026-10-20T10:00:00Z",
        },
        {
            "status": "VERIFIED",
            "file_path": "val/FAKE/workplace_fake_00001.jpg",
            "filename": "workplace_fake_00001.jpg",
            "split": "val",
            "label": "FAKE",
            "class_idx": 0,
            "sha256": "raw_payload_hash_222222",
            "raw_payload_sha256": "raw_payload_hash_222222",
            "url": "https://datasets-server.hf.co/gen1.jpg",
            "title": "AI Worker",
            "source": "OpenFake (model-x)",
            "license": "CC-BY-NC",
            "author_or_generator": "model-x",
            "timestamp": "2026-10-20T11:00:00Z",
        },
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for le in log_entries:
            f.write(json.dumps(le) + "\n")

    # Create initial ledger containing file3 as historical entry without URL
    ledger_file = dataset_dir / "provenance_ledger.json"
    initial_ledger = {
        "dataset_root": str(dataset_dir).replace("\\", "/"),
        "quarantine_count": 1,
        "verified_records": [
            {
                "file_path": "test/REAL/local_real_legacy.jpg",
                "filename": "local_real_legacy.jpg",
                "split": "test",
                "label": "REAL",
                "class_idx": 1,
                "sha256": compute_sha256_file(file3),
                "source": "OpenFake / Verified Archive",
                "generator": "Unknown",
                "license": "CC0 / Open Research",
                "status": "VERIFIED_ACTIVE",
            }
        ],
        "quarantined_records": [
            {
                "file_path": "quarantine/quarantined_sample.jpg",
                "reason": "Test quarantine reason",
                "sha256": "sample_q_hash",
            }
        ],
        "verified_count": 1,
    }
    with open(ledger_file, "w", encoding="utf-8") as f:
        json.dump(initial_ledger, f, indent=2)

    return dataset_dir


class TestProvenanceLedgerBackfill:

    def test_reconciliation_and_hash_separation(self, temp_dataset_env):
        """Verifies full reconciliation with raw_payload_sha256 and final_disk_sha256 separation."""
        res = run_backfill(dry_run=False, dataset_root=temp_dataset_env)
        assert res["status"] == "PASS"
        assert res["verified_count"] == 3
        assert res["full_provenance_count"] == 2
        assert res["incomplete_provenance_count"] == 1

        ledger_path = temp_dataset_env / "provenance_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        records = {r["file_path"]: r for r in ledger["verified_records"]}

        # 1. file1: check hashes
        r1 = records["train/REAL/workplace_real_00001.jpg"]
        actual_disk_1 = compute_sha256_file(temp_dataset_env / "train/REAL/workplace_real_00001.jpg")
        assert r1["final_disk_sha256"] == actual_disk_1
        assert r1["raw_payload_sha256"] == "raw_payload_hash_111111"
        assert r1["raw_payload_sha256"] != r1["final_disk_sha256"]
        assert r1["url"] == "https://upload.wikimedia.org/office1.jpg"
        assert r1["source"] == "Wikimedia Commons"

        # 2. file3 (historical baseline): missing fields are explicitly unknown
        r3 = records["test/REAL/local_real_legacy.jpg"]
        actual_disk_3 = compute_sha256_file(temp_dataset_env / "test/REAL/local_real_legacy.jpg")
        assert r3["final_disk_sha256"] == actual_disk_3
        assert r3["raw_payload_sha256"] == "unknown"
        assert r3["url"] == "unknown"
        assert r3["timestamp"] == "unknown"

    def test_idempotent_repeated_execution(self, temp_dataset_env):
        """Verifies running backfill multiple times maintains exact record counts and no duplicates."""
        res1 = run_backfill(dry_run=False, dataset_root=temp_dataset_env)
        assert res1["status"] == "PASS"
        assert res1["verified_count"] == 3

        res2 = run_backfill(dry_run=False, dataset_root=temp_dataset_env)
        assert res2["status"] == "PASS"
        assert res2["verified_count"] == 3

        ledger_path = temp_dataset_env / "provenance_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        assert len(ledger["verified_records"]) == 3
        assert ledger["verified_count"] == 3

    def test_path_normalization(self, temp_dataset_env):
        """Verifies all paths are normalized with forward slashes across platforms."""
        run_backfill(dry_run=False, dataset_root=temp_dataset_env)
        ledger_path = temp_dataset_env / "provenance_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)

        for r in ledger["verified_records"]:
            assert "\\" not in r["file_path"]
            assert r["file_path"].startswith(("train/", "val/", "test/"))

    def test_quarantine_records_preserved(self, temp_dataset_env):
        """Verifies quarantine records and quarantine count are untouched."""
        res = run_backfill(dry_run=False, dataset_root=temp_dataset_env)
        assert res["quarantine_count"] == 1

        ledger_path = temp_dataset_env / "provenance_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        assert len(ledger["quarantined_records"]) == 1
        assert ledger["quarantined_records"][0]["sha256"] == "sample_q_hash"

    def test_conflicting_metadata_detected(self, temp_dataset_env):
        """Verifies conflicting class/split in metadata blocks migration."""
        log_file = temp_dataset_env / "acquisition_log.jsonl"
        # Overwrite log entry with conflicting label
        with open(log_file, "a", encoding="utf-8") as f:
            conflict_entry = {
                "status": "VERIFIED",
                "file_path": "train/REAL/workplace_real_00001.jpg",
                "filename": "workplace_real_00001.jpg",
                "split": "train",
                "label": "FAKE", # Contradictory label
            }
            # Append conflicting entry
            f.write(json.dumps(conflict_entry) + "\n")

        res = run_backfill(dry_run=True, dataset_root=temp_dataset_env)
        assert res["status"] == "BLOCKED"
        assert len(res["conflicts"]) > 0
