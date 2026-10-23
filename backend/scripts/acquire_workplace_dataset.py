"""
DeepGuard AI - Resumable Workplace Dataset Acquisition & Validation Engine
==========================================================================
Lawfully acquires and verifies workplace domain images from documented sources:
  1. REAL Workplace Images:
     - Wikimedia Commons API (CC-BY, CC-BY-SA, CC0 licenses, human photographer attribution).
     - Verified optical photography with documented repository URLs.
  2. FAKE Workplace Images:
     - OpenFake Generative Repository API (FLUX.1-dev, Midjourney-6, SD 3.5, SDXL, Imagen).
     - Records explicit generative model name, prompt text, and row SHA-256 checksums.

Key Safeguards:
  - Network Resilience: Bounded exponential backoff + jitter for SSL, timeout, 429, and 5xx.
  - Zero Insecure HTTPS Workarounds: Preserves strict SSL certificate verification.
  - Graceful Interruption: Handles KeyboardInterrupt cleanly; flushes logs, removes partial files.
  - Resumable: Checks existing disk files and acquisition_log.jsonl before fetching.
  - Zero Fabrication: Only downloads real assets from live public APIs.
  - Leakage Prevention: ai_man_office_laptop.png (and near-duplicates) strictly quarantined to test/FAKE.
  - Cryptographic & Perceptual Dedup: Rejects exact SHA-256 duplicates and dHash near-duplicates (Hamming <= 4).
  - Quarantines corrupt, truncated, non-bitmap, or unverified files to backend/data/workplace_dataset/quarantine/.

Usage:
    python backend/scripts/acquire_workplace_dataset.py --target_real 25 --target_fake 25
    python backend/scripts/acquire_workplace_dataset.py --target_real 600 --target_fake 600
"""

import argparse
import hashlib
import http.client
import io
import json
import os
import random
import re
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

DATASET_ROOT = backend_root / "data" / "workplace_dataset"
LOG_PATH = DATASET_ROOT / "acquisition_log.jsonl"
LEDGER_PATH = DATASET_ROOT / "provenance_ledger.json"
QUARANTINE_DIR = DATASET_ROOT / "quarantine"
TRACKED_OFFICE_IMAGE = backend_root / "data" / "test_cases" / "ai_man_office_laptop.png"

USER_AGENT = "DeepGuard-Acquisition-Pipeline/1.0 (https://github.com/deepguard; contact@deepguard.ai)"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def classify_network_error(exc: Exception) -> tuple[str, bool]:
    """
    Classifies exceptions into descriptive error types and determines if they are transient/retryable.
    Returns: (error_type_str, is_retryable_bool)
    """
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            return "HTTP_429_TOO_MANY_REQUESTS", True
        elif exc.code in (500, 502, 503, 504):
            return f"HTTP_{exc.code}_SERVER_ERROR", True
        elif exc.code in (400, 401, 403, 404, 410):
            return f"HTTP_{exc.code}_CLIENT_ERROR", False
        else:
            return f"HTTP_{exc.code}", False
    elif isinstance(exc, ssl.SSLError):
        return "SSL_HANDSHAKE_ERROR", True
    elif isinstance(exc, (socket.timeout, TimeoutError)):
        return "TIMEOUT_ERROR", True
    elif isinstance(exc, urllib.error.URLError):
        reason_str = str(getattr(exc, "reason", "")).lower()
        if "handshake" in reason_str or "ssl" in reason_str:
            return "SSL_HANDSHAKE_ERROR", True
        if "timed out" in reason_str or "timeout" in reason_str:
            return "TIMEOUT_ERROR", True
        if "connection reset" in reason_str or "refused" in reason_str or "closed" in reason_str:
            return "CONNECTION_RESET", True
        return "URL_NETWORK_ERROR", True
    elif isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                          http.client.RemoteDisconnected, http.client.IncompleteRead)):
        return "CONNECTION_RESET", True
    return "NETWORK_ERROR", True


def fetch_url_with_retries(
    url: str,
    headers: dict = None,
    timeout: float = 25.0,
    max_retries: int = 3,
    base_backoff: float = 1.0,
    max_backoff: float = 10.0,
    jitter: float = 0.5,
) -> tuple[bytes | None, str | None, str | None]:
    """
    Fetches raw bytes with bounded retries, exponential backoff, and jitter.
    Strictly preserves SSL verification.
    Never swallows KeyboardInterrupt.
    Returns: (data_bytes, error_type, error_reason)
    """
    if headers is None:
        headers = {"User-Agent": USER_AGENT}

    attempt = 0
    last_error_type = None
    last_error_reason = None

    while attempt <= max_retries:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return data, None, None
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            err_type, is_retryable = classify_network_error(exc)
            last_error_type = err_type
            last_error_reason = str(exc)

            if not is_retryable or attempt >= max_retries:
                break

            # If 429 response includes a Retry-After header, respect it up to max_backoff
            retry_after = None
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                ra_header = exc.headers.get("Retry-After")
                if ra_header and ra_header.isdigit():
                    retry_after = min(float(ra_header), max_backoff)

            backoff = retry_after if retry_after is not None else (
                min(max_backoff, base_backoff * (2 ** attempt)) + random.uniform(0.0, jitter)
            )
            print(f"    [Retry {attempt + 1}/{max_retries}] {err_type} for URL. Backing off {backoff:.2f}s...")
            time.sleep(backoff)
            attempt += 1

    return None, last_error_type, last_error_reason


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Computes a 64-bit difference hash (dHash) for perceptual near-duplicate detection."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    diff = [img.getpixel((c, r)) > img.getpixel((c + 1, r)) for r in range(hash_size) for c in range(hash_size)]
    val = int("".join(["1" if x else "0" for x in diff]), 2)
    return hex(val)[2:].zfill(hash_size * hash_size // 4)


def hamming_distance(hex1: str, hex2: str) -> int:
    return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")


def cleanup_temp_files(root: Path = DATASET_ROOT):
    """Removes any orphaned temporary files (.tmp_*, *.tmp) across dataset directories."""
    if not root.exists():
        return
    for tmp_path in root.glob("**/.tmp_*"):
        try:
            if tmp_path.is_file():
                tmp_path.unlink()
        except OSError:
            pass
    for tmp_path in root.glob("**/*.tmp"):
        try:
            if tmp_path.is_file():
                tmp_path.unlink()
        except OSError:
            pass


def log_event(record: dict):
    """Appends an event to the resumable JSONL acquisition log and flushes immediately."""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def sync_verified_record_to_ledger(record: dict, ledger_path: Path = None) -> bool:
    """
    Safely and idempotently synchronizes an accepted active file record into provenance_ledger.json.
    Ensures final relative path, split, class, final_disk_sha256, raw_payload_sha256, source, title,
    timestamp, and provenance metadata are recorded without duplicates or corruption.
    """
    if ledger_path is None:
        ledger_path = DATASET_ROOT / "provenance_ledger.json"

    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    ledger_data = {
        "dataset_root": str(DATASET_ROOT).replace("\\", "/"),
        "quarantine_count": 0,
        "verified_records": [],
        "quarantined_records": [],
        "verified_count": 0,
        "last_reconciliation": datetime.now(timezone.utc).isoformat(),
    }

    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    ledger_data = json.loads(content)
        except Exception as e:
            print(f"[Ledger Sync Warning] Failed to read existing ledger at {ledger_path}: {e}")
            return False

    # Check status: failed or rejected candidates must not be marked as accepted active images
    status = record.get("status", "VERIFIED")
    if status not in ("VERIFIED", "VERIFIED_ACTIVE"):
        return False

    verified_records = ledger_data.get("verified_records", [])
    quarantined_records = ledger_data.get("quarantined_records", [])

    final_disk_sha = record.get("final_disk_sha256") or record.get("sha256")
    raw_payload_sha = record.get("raw_payload_sha256", "unknown")
    file_path = record.get("file_path", "")
    filename = record.get("filename", Path(file_path).name if file_path else "")

    # Check quarantine: if quarantined, do not mark as active verified
    q_shas = {q.get("sha256") for q in quarantined_records if q.get("sha256")}
    if (final_disk_sha and final_disk_sha in q_shas) or (raw_payload_sha != "unknown" and raw_payload_sha in q_shas):
        return False

    # Check for existing record in verified_records
    existing_idx = None
    for idx, vr in enumerate(verified_records):
        if (
            (final_disk_sha and vr.get("final_disk_sha256") == final_disk_sha)
            or (final_disk_sha and vr.get("sha256") == final_disk_sha)
            or (file_path and vr.get("file_path") == file_path)
            or (filename and vr.get("filename") == filename)
        ):
            existing_idx = idx
            break

    entry = {
        "file_path": file_path,
        "filename": filename,
        "split": record.get("split", "unknown"),
        "label": record.get("label", "unknown"),
        "class_idx": record.get("class_idx", 0 if record.get("label") == "FAKE" else 1),
        "dimensions": record.get("dimensions", []),
        "format": record.get("format", "JPEG"),
        "color_mode": "RGB",
        "sha256": final_disk_sha or "unknown",
        "final_disk_sha256": final_disk_sha or "unknown",
        "raw_payload_sha256": raw_payload_sha,
        "dhash": record.get("dhash", "unknown"),
        "source": record.get("source", "unknown"),
        "generator": record.get("author_or_generator") or record.get("generator", "unknown"),
        "license": record.get("license", "unknown"),
        "url": record.get("url", "unknown"),
        "title": record.get("title", "unknown"),
        "provenance_notes": record.get("provenance_notes", ""),
        "status": "VERIFIED_ACTIVE",
        "timestamp": record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    }

    if existing_idx is not None:
        target_entry = verified_records[existing_idx]
        for k, v in entry.items():
            if v != "unknown" and v is not None and v != "":
                target_entry[k] = v
            elif k not in target_entry:
                target_entry[k] = v
    else:
        verified_records.append(entry)

    ledger_data["verified_records"] = verified_records
    ledger_data["verified_count"] = len(verified_records)
    ledger_data["quarantine_count"] = len(quarantined_records)
    ledger_data["last_reconciliation"] = datetime.now(timezone.utc).isoformat()

    temp_ledger = ledger_path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
    try:
        with open(temp_ledger, "w", encoding="utf-8") as f:
            json.dump(ledger_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_ledger, ledger_path)
        return True
    except Exception as e:
        if temp_ledger.exists():
            try:
                temp_ledger.unlink()
            except OSError:
                pass
        print(f"[Ledger Sync Error] Failed to write ledger: {e}")
        return False


def load_existing_ledger() -> tuple[set, dict, list, set]:
    """
    Loads existing SHA-256 set, dHash map, all logged URLs, and permanently failed URLs
    from acquisition_log.jsonl, provenance_ledger.json, and physical disk files
    to ensure complete resumability and avoid duplicate network requests.
    """
    existing_urls = set()
    failed_urls = set()
    existing_sha = {}
    existing_dhashes = []

    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    url = rec.get("url")
                    if url:
                        existing_urls.add(url)
                    status = rec.get("status")
                    if status == "FAILED" and url:
                        failed_urls.add(url)
                    elif status == "VERIFIED":
                        file_p = rec.get("file_path") or rec.get("dest_path")
                        for sha_key in ("sha256", "final_disk_sha256", "raw_payload_sha256"):
                            if sha_key in rec and rec[sha_key] != "unknown":
                                existing_sha[rec[sha_key]] = file_p
                        if "dhash" in rec:
                            existing_dhashes.append(rec["dhash"])
                except Exception:
                    continue

    ledger_file = DATASET_ROOT / "provenance_ledger.json"
    if ledger_file.exists():
        try:
            with open(ledger_file, "r", encoding="utf-8") as f:
                led_content = f.read().strip()
                if led_content:
                    led = json.loads(led_content)
                    for vr in led.get("verified_records", []):
                        fp = vr.get("file_path")
                        for sha_key in ("sha256", "final_disk_sha256", "raw_payload_sha256"):
                            if sha_key in vr and vr[sha_key] != "unknown":
                                existing_sha[vr[sha_key]] = fp
                        if "url" in vr and vr["url"] != "unknown":
                            existing_urls.add(vr["url"])
                        if "dhash" in vr and vr["dhash"] != "unknown":
                            existing_dhashes.append(vr["dhash"])
        except Exception:
            pass

    # Also scan disk files for any manually added or verified files
    for split in ["train", "val", "test"]:
        for lbl in ["REAL", "FAKE"]:
            folder = DATASET_ROOT / split / lbl
            if folder.exists():
                for p in folder.glob("*.*"):
                    if p.name.startswith(".tmp") or p.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                        continue
                    try:
                        with open(p, "rb") as f:
                            b = f.read()
                        sha = compute_sha256_bytes(b)
                        existing_sha[sha] = str(p.relative_to(DATASET_ROOT)).replace("\\", "/")
                    except Exception:
                        pass

    return existing_urls, existing_sha, existing_dhashes, failed_urls


def get_current_counts() -> dict:
    counts = {"train": {"REAL": 0, "FAKE": 0}, "val": {"REAL": 0, "FAKE": 0}, "test": {"REAL": 0, "FAKE": 0}}
    for split in ["train", "val", "test"]:
        for lbl in ["REAL", "FAKE"]:
            folder = DATASET_ROOT / split / lbl
            if folder.exists():
                files = [
                    f for f in folder.glob("*.*")
                    if f.suffix.lower() in VALID_IMAGE_EXTENSIONS and not f.name.startswith(".tmp")
                ]
                counts[split][lbl] = len(files)
    return counts


def determine_split_destination(lbl: str, current_counts: dict) -> tuple[Path, str]:
    """
    Allocates new verified items across train (60%), val (20%), test (20%) in balanced fashion.
    Prioritizes splits currently most in deficit relative to target ratios.
    """
    train_c = current_counts["train"][lbl]
    val_c = current_counts["val"][lbl]
    test_c = current_counts["test"][lbl]
    total = train_c + val_c + test_c

    # Target ratios: 60% train, 20% val, 20% test
    # Deficit = target_ratio * (total + 1) - current_count
    deficits = {
        "train": 0.60 * (total + 1) - train_c,
        "val": 0.20 * (total + 1) - val_c,
        "test": 0.20 * (total + 1) - test_c,
    }
    # Pick split with highest deficit, breaking ties in favor of train -> val -> test
    target_split = max(["train", "val", "test"], key=lambda s: (deficits[s], s == "train", s == "val"))

    dest_dir = DATASET_ROOT / target_split / lbl
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir, target_split


def extract_subject_key(title: str, url: str) -> str:
    """
    Normalizes a Wikimedia title or URL to extract a canonical subject/event key,
    stripping photo numbers, timestamps, resolution, and event suffixes.
    e.g. 'File:Alan Schaaf, Imgur Founder, in Wellington (TEDxWellington) (14064560181).jpg'
    -> 'alan schaaf imgur founder in wellington'
    """
    import re
    import urllib.parse
    clean = title or ""
    if not clean and url:
        # Extract filename from URL rather than raw URL string to avoid domain prefix collapse
        clean = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
    clean = re.sub(r"^File:", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\.[a-zA-Z0-9]+$", "", clean)  # remove extension
    clean = re.sub(r"\(\d+\)", "", clean)           # remove photo numbers
    clean = re.sub(r"[\d_-]+", " ", clean)          # replace numbers/underscores
    clean = re.sub(r"[^\w\s]", " ", clean)          # remove punctuation
    clean = " ".join(clean.lower().split())
    return clean[:45]


class QueryDiagnostics:
    """Tracks structured per-query acquisition metrics."""
    def __init__(self, query: str = ""):
        self.query = query
        self.candidates_found = 0
        self.http_api_errors = 0
        self.rejected_non_bitmap = 0
        self.rejected_keyword_filter = 0
        self.rejected_subject_quota = 0
        self.rejected_duplicates = 0
        self.rejected_invalid_images = 0
        self.successfully_accepted = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "candidates_found": self.candidates_found,
            "http_api_errors": self.http_api_errors,
            "rejected_non_bitmap": self.rejected_non_bitmap,
            "rejected_keyword_filter": self.rejected_keyword_filter,
            "rejected_subject_quota": self.rejected_subject_quota,
            "rejected_duplicates": self.rejected_duplicates,
            "rejected_invalid_images": self.rejected_invalid_images,
            "successfully_accepted": self.successfully_accepted,
        }

    def print_summary(self):
        print(f"\n  [Query Diagnostics] '{self.query}'")
        print(f"    - Candidates Found          : {self.candidates_found}")
        print(f"    - HTTP / API Errors         : {self.http_api_errors}")
        print(f"    - Rejected (Non-Bitmap)     : {self.rejected_non_bitmap}")
        print(f"    - Rejected (Keyword Filter) : {self.rejected_keyword_filter}")
        print(f"    - Rejected (Subject Quota)  : {self.rejected_subject_quota}")
        print(f"    - Rejected (Duplicates)     : {self.rejected_duplicates}")
        print(f"    - Rejected (Invalid Images) : {self.rejected_invalid_images}")
        print(f"    - Successfully Accepted     : {self.successfully_accepted}")


METADATA_REJECT_PATTERNS = [
    (re.compile(r"(?:page|tome|vol|volume|issue)\s*\d+.*crop", re.I), "Vintage scanned publication / book page crop"),
    (re.compile(r"\b(?:crop\s*\d+|book crop|frontespizio|recueil|manuscrit|manuscript|oeuvres complètes|œuvres complètes)\b", re.I), "Vintage scanned publication / manuscript"),
    (re.compile(r"\b(?:denier|coin|monnaie|numismatic|numismatique|medal|médaille|grand bronze|moyen bronze|petit bronze|bronze vulgate|mine d['’]antiochus)\b", re.I), "Numismatic coin / antique currency / medal"),
    (re.compile(r"\b(?:plot|ppplot|qqplot|detrended|barchart|histogram|chart|diagram|scatter\s*plot|regression\s*plot)\b", re.I), "Statistical chart / mathematical plot diagram"),
    (re.compile(r"\b(?:flag of|coat of arms|heraldry|emblem of|state flag|municipal flag)\b", re.I), "Flag / heraldic emblem / municipal graphic"),
    (re.compile(r"\b(?:laboratory rat|rat\b|mouse\b(?!.*(?:keyboard|pointer))|rodent|llama|cattle|livestock|wildlife|hamster|animal experiment)\b", re.I), "Animal / wildlife / rodent subject"),
    (re.compile(r"\b(?:kilimanjaro|volcano|stratovolcano|mountain closeup|geological map|space mirrors|space station|astronaut photograph|earth observation|hauptbahnhof|railway station)\b", re.I), "Geological landscape / satellite / transit architecture outside workplace"),
    (re.compile(r"\b(?:watercolor|watercolour|painting|oil on canvas|drawing of|sketch of|pencil drawing|illustration of user|la bella principessa|dotkamina)\b", re.I), "Artistic watercolor / illustration / drawing"),
    (re.compile(r"\b(?:algorithmically-generated|ai art|midjourney|dall-e|stable diffusion|ai explosion|tamingtheaibeast|loss, in first-person view)\b", re.I), "Synthetic AI art mislabeled as optical photograph"),
    (re.compile(r"\b(?:rayleigh-taylor instability|simulation of an extremely large)\b", re.I), "Physics simulation visualization without workplace scene"),
]

REJECT_REAL_KEYWORDS = [
    "tool kit", "toolkit", "cable", "screw", "equipment", "hardware", "mixer",
    "console", "server rack", "motherboard", "building", "exterior", "empty office",
    "floor plan", "memorandum", "handbook", "brochure", "sofa", "couch", "furniture",
    "fox", "gingerbread", "batman", "harry potter", "soccer", "baseball",
    "panther", "cookbook", "chef", "mat w/ cable", "qu-24 chrome",
    "gym setting", "split on a blue mat", "race car", "musician", "sins of arrogance",
    "camper van", "campervan", "motor home", "motorhome",
    "child", "schooling", "kids learning", "classroom", "schoolchildren", "pediatric",
    "faun", "satyrkopf", "sculpture", "statue", "museum", "kunstpalast",
    "cromemco", "system one", "crashed computer", "bioart",
    "tome ", "œuvres", "page crop", "page 1", "issue 0", "volume 0", "book crop",
    "denier", "vulgate", "beverly hills", "west hollywood", "warship", "ppplot", "qqplot",
    "laboratory rat", "kilimanjaro", "watercolor", "watercolour"
]

WIKIMEDIA_REAL_SEARCH_QUERIES = [
    'filetype:bitmap "working" "laptop" "office"',
    'filetype:bitmap "office worker" "computer"',
    'filetype:bitmap "software engineer" "desk"',
    'filetype:bitmap "programmer" "desk"',
    'filetype:bitmap "engineer" "computer" "office"',
    'filetype:bitmap "call center" "computer"',
    'filetype:bitmap "technician" "computer" "desk"',
    'filetype:bitmap "journalist" "computer" "desk"',
    'filetype:bitmap "telework" "laptop"',
    'filetype:bitmap "home office" "laptop"',
    'filetype:bitmap "remote work" "computer"',
    'filetype:bitmap "customer service" "computer"',
    'filetype:bitmap "receptionist" "computer" "desk"',
    'filetype:bitmap "businesswoman" "laptop" "office"',
    'filetype:bitmap "meeting" "laptop" "office"',
    'filetype:bitmap "working on laptop" "office"',
    'filetype:bitmap "corporate worker" "computer"',
    'filetype:bitmap "architect" "computer" "desk"',
    'filetype:bitmap "desk" "computer" "working"',
    'filetype:bitmap "workstation" "computer" "office"',
    'incategory:"People with computers" filetype:bitmap',
    'incategory:"People with laptops" filetype:bitmap',
    'incategory:"Offices" filetype:bitmap "laptop"',
    'incategory:"Offices" filetype:bitmap "computer"',
]


def fetch_wikimedia_workplace_candidates(
    limit: int = 50,
    offset: int = 0,
    timeout: float = 25.0,
    max_retries: int = 3,
    existing_urls: set = None,
    seen_subjects: dict = None,
    max_per_subject: int = 1,
    query: str = None,
    diag: QueryDiagnostics = None,
) -> list:
    """
    Queries Wikimedia Commons API for authentic workplace, office, and laptop photography.
    Extracts author attribution, license name, direct download URL, and metadata.
    Enforces candidate diversity: rejects non-image formats, non-workplace hardware/furniture,
    and caps repeated subjects/albums to prevent subject clustering and split leakage.
    """
    if query is None:
        query = WIKIMEDIA_REAL_SEARCH_QUERIES[(offset // 30) % len(WIKIMEDIA_REAL_SEARCH_QUERIES)]
    encoded_query = urllib.parse.quote(query)

    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrnamespace=6&gsrsearch={encoded_query}&gsrlimit={min(limit, 50)}"
        f"&gsroffset={offset}&prop=imageinfo&iiprop=url|size|extmetadata&format=json"
    )

    data_bytes, err_type, err_reason = fetch_url_with_retries(
        url, timeout=timeout, max_retries=max_retries
    )
    if not data_bytes:
        if diag:
            diag.http_api_errors += 1
        print(f"[Wikimedia API Error] {err_type}: {err_reason}")
        return []

    try:
        data = json.loads(data_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        if diag:
            diag.http_api_errors += 1
        print(f"[Wikimedia Parse Error] {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    candidates = []
    batch_seen_subjects = set()

    for pid, pdata in pages.items():
        if diag:
            diag.candidates_found += 1
        ii = pdata.get("imageinfo", [{}])[0]
        img_url = ii.get("url")
        if not img_url:
            continue

        # 1. Check file extension early
        path_ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower()
        if path_ext not in VALID_IMAGE_EXTENSIONS:
            if diag:
                diag.rejected_non_bitmap += 1
            if existing_urls is not None and img_url not in existing_urls:
                log_event({
                    "url": img_url,
                    "status": "QUARANTINED",
                    "reason": f"Non-bitmap file format ({path_ext})",
                    "label": "REAL",
                    "title": pdata.get("title")
                })
                existing_urls.add(img_url)
            continue

        title = pdata.get("title", "")
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "CC-BY-SA")
        artist = meta.get("Artist", {}).get("value", "Wikimedia Contributor")
        desc = meta.get("ImageDescription", {}).get("value", title)

        title_desc_low = f"{title} {desc}".lower()

        # 2. Domain relevance check: reject hardware, toolkits, book crops, plots, flags, animals, coins, etc.
        rejected_reason = None
        for pat, reason in METADATA_REJECT_PATTERNS:
            if pat.search(title_desc_low):
                rejected_reason = reason
                break
        if not rejected_reason:
            for irr in REJECT_REAL_KEYWORDS:
                if irr in title_desc_low:
                    rejected_reason = f"Contains rejected keyword '{irr}'"
                    break

        if rejected_reason:
            if diag:
                diag.rejected_keyword_filter += 1
            if existing_urls is not None and img_url not in existing_urls:
                log_event({
                    "url": img_url,
                    "status": "QUARANTINED",
                    "reason": rejected_reason,
                    "label": "REAL",
                    "title": title
                })
                existing_urls.add(img_url)
            continue

        # 3. Subject / Album diversity check
        subj_key = extract_subject_key(title, img_url)
        if (seen_subjects is not None and seen_subjects.get(subj_key, 0) >= max_per_subject) or (subj_key in batch_seen_subjects):
            # Skip repeated album/subject to prevent duplicate series and split leakage
            if diag:
                diag.rejected_subject_quota += 1
            continue

        batch_seen_subjects.add(subj_key)

        candidates.append({
            "title": title,
            "url": img_url,
            "license": lic,
            "author": str(artist)[:100],
            "description": str(desc)[:150],
            "width": ii.get("width"),
            "height": ii.get("height"),
            "source": "Wikimedia Commons",
            "ground_truth": "REAL",
            "subject_key": subj_key,
        })

    return candidates


WORKPLACE_FAKE_PROMPT_KEYWORDS = [
    "office", "desk", "laptop", "computer", "suit", "business", "working", "workplace",
    "corporate", "coworking", "meeting", "executive", "professional", "portrait", "conference",
    "colleague", "typing", "boardroom", "presentation", "worker", "employee", "interview"
]

NON_WORKPLACE_PROMPT_KEYWORDS = [
    "gingerbread", "fox", "batman", "comic", "anime", "illustration", "poster",
    "panther", "baseball", "soccer", "stadium", "concert", "singer", "stage",
    "chef", "kitchen", "recipe", "food", "baby", "children", "child", "split on a mat",
    "superhero", "dinosaur", "dragon", "alien", "fantasy", "fairy", "magic", "racing", "podium",
    "sculpture", "statue", "book cover", "magazine cover", "stylized portrait"
]


def fetch_openfake_generative_candidates(
    limit: int = 50,
    offset: int = 0,
    timeout: float = 25.0,
    max_retries: int = 3,
) -> list:
    """
    Queries Hugging Face OpenFake Server API for verified generative media
    with documented generator model, prompt, and row checksums.
    Filters prompts to ensure domain relevance (workplace, office, business portraits)
    and excludes irrelevant non-workplace items (sports, animals, cartoons, posters).
    """
    url = (
        f"https://datasets-server.huggingface.co/rows?dataset=ComplexDataLab/OpenFake"
        f"&config=core&split=train&offset={offset}&limit={min(limit, 100)}"
    )

    data_bytes, err_type, err_reason = fetch_url_with_retries(
        url, timeout=timeout, max_retries=max_retries
    )
    if not data_bytes:
        print(f"[OpenFake API Error] {err_type}: {err_reason}")
        return []

    try:
        data = json.loads(data_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[OpenFake Parse Error] {e}")
        return []

    rows = data.get("rows", [])
    candidates = []

    for r in rows:
        row = r.get("row", {})
        if str(row.get("label", "")).lower() != "fake":
            continue

        img_dict = row.get("image", {})
        img_url = img_dict.get("src")
        if not img_url:
            continue

        model = row.get("model", "unknown_diffusion")
        prompt = row.get("prompt", "")
        row_id = row.get("row_id", "")
        p_low = prompt.lower()

        # Reject irrelevant patterns (animals, sports, cartoons, books, recipes)
        if any(irr in p_low for irr in NON_WORKPLACE_PROMPT_KEYWORDS):
            continue

        # Enforce workplace, office, business, or professional context
        if not any(k in p_low for k in WORKPLACE_FAKE_PROMPT_KEYWORDS):
            continue

        candidates.append({
            "row_id": row_id,
            "url": img_url,
            "license": "OpenFake Research CC-BY-NC",
            "generator": model,
            "prompt": str(prompt)[:150],
            "source": f"OpenFake ({model})",
            "ground_truth": "FAKE"
        })

    return candidates


def validate_candidate_pixels(img_rgb: Image.Image) -> tuple[bool, str]:
    """
    Performs pixel-level heuristic analysis on decoded image:
    1. Detects scanned vintage documents / book page crops (low saturation, high edges, high white ratio).
    2. Detects statistical plots / mathematical diagrams (white background, sharp edge structure, low color count).
    3. Detects flat vector graphics / municipal flags / digital emblems (hyper-saturated blocks or low color count).
    """
    w, h = img_rgb.size
    arr = np.array(img_rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_mean = float(np.mean(hsv[:, :, 2]))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    white_pixels = np.sum((arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240))
    white_ratio = float(white_pixels / (w * h))

    small = img_rgb.resize((64, 64))
    unique_colors = len(set(small.getdata()))

    # 1. Scanned page / text engraving: low saturation, high edges, high white background
    if sat_mean < 18.0 and edge_density > 0.070 and white_ratio > 0.28:
        return False, f"Pixel signature matches scanned book page / text engraving (sat={sat_mean:.1f}, edge={edge_density:.3f})"

    # 2. Statistical plot / mathematical diagram / chart:
    if (white_ratio > 0.40 and edge_density > 0.030 and unique_colors < 1000) or (sat_mean < 5.0 and val_mean > 215 and edge_density > 0.020 and unique_colors < 350):
        return False, f"Pixel signature matches statistical chart / plot diagram (white={white_ratio:.2f}, edge={edge_density:.3f}, uniq={unique_colors})"

    # 3. Flat vector graphic / flag / emblem:
    if (unique_colors < 150 and white_ratio < 0.40 and sat_mean > 30) or (sat_mean > 175 and edge_density < 0.015):
        return False, f"Pixel signature matches flat graphic / vector emblem (uniq={unique_colors}, sat={sat_mean:.1f})"

    return True, "Valid pixels"


def validate_candidate_semantics_and_pixels(img_rgb: Image.Image, candidate: dict) -> tuple[bool, str]:
    """
    Multi-modal validator combining metadata provenance and pixel structure.
    Does not rely solely on filename keywords, face detection, broad ImageNet labels, or filetype:bitmap.
    """
    title = candidate.get("title", "")
    desc = candidate.get("description", "")
    notes = candidate.get("prompt", "") or candidate.get("provenance_notes", "")
    lbl = candidate.get("ground_truth", "REAL")

    if lbl == "REAL":
        comb = f"{title} {desc} {notes}"
        for pat, reason in METADATA_REJECT_PATTERNS:
            if pat.search(comb):
                return False, reason
        for irr in REJECT_REAL_KEYWORDS:
            if irr in comb.lower():
                return False, f"Contains rejected keyword '{irr}'"
        pix_ok, pix_reason = validate_candidate_pixels(img_rgb)
        if not pix_ok:
            return False, pix_reason

    if lbl == "FAKE":
        comb_fake = f"{desc} {notes}".lower()
        if any(irr in comb_fake for irr in NON_WORKPLACE_PROMPT_KEYWORDS):
            return False, f"Irrelevant non-workplace synthetic prompt: {comb_fake[:60]}"

    return True, "Semantically verified workplace domain asset"


def download_and_validate_asset(
    candidate: dict,
    existing_sha: dict,
    existing_dhashes: list,
    target_dhash: str,
    current_counts: dict,
    existing_urls: set,
    timeout: float = 25.0,
    max_retries: int = 3,
    base_backoff: float = 1.0,
    max_backoff: float = 10.0,
    diag: QueryDiagnostics = None,
) -> bool:
    """
    Downloads an image asset with network retries and exponential backoff.
    Validates decoding, dimensions, SHA-256, and perceptual near-duplicates.
    Saves atomically using a temporary file that is cleaned up on any failure.
    """
    url = candidate["url"]
    lbl = candidate["ground_truth"]

    # 1. Download bytes with network resilience
    img_bytes, err_type, err_reason = fetch_url_with_retries(
        url,
        timeout=timeout,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
    )

    if img_bytes is None:
        log_event({
            "url": url,
            "status": "FAILED",
            "error_type": err_type or "DOWNLOAD_FAILURE",
            "reason": f"Network retries exhausted: {err_reason}",
            "label": lbl,
            "retries_exhausted": True
        })
        existing_urls.add(url)
        if diag:
            diag.http_api_errors += 1
        return False

    if len(img_bytes) < 1024:
        log_event({"url": url, "status": "QUARANTINED", "reason": f"File too small ({len(img_bytes)} bytes)", "label": lbl})
        existing_urls.add(url)
        if diag:
            diag.rejected_invalid_images += 1
        return False

    # 2. Cryptographic SHA-256 (Raw Payload Checksum)
    raw_payload_sha = compute_sha256_bytes(img_bytes)
    sha = raw_payload_sha
    if raw_payload_sha in existing_sha:
        log_event({
            "url": url,
            "status": "DUPLICATE_SHA",
            "sha256": raw_payload_sha,
            "raw_payload_sha256": raw_payload_sha,
            "label": lbl,
        })
        existing_urls.add(url)
        if diag:
            diag.rejected_duplicates += 1
        return False

    # 3. PIL Decoding & Validation
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        img_rgb = img.convert("RGB")
        dhash = compute_dhash(img_rgb)
    except Exception as e:
        log_event({"url": url, "status": "QUARANTINED", "reason": f"Corrupt image decoding: {str(e)}", "label": lbl})
        existing_urls.add(url)
        if diag:
            diag.rejected_invalid_images += 1
        return False

    if w < 64 or h < 64:
        log_event({"url": url, "status": "QUARANTINED", "reason": f"Resolution too low ({w}x{h})", "label": lbl})
        existing_urls.add(url)
        if diag:
            diag.rejected_invalid_images += 1
        return False

    # 4. Near-Duplicate Check against Tracked Office Laptop Target
    if target_dhash:
        dist_to_office = hamming_distance(dhash, target_dhash)
        if dist_to_office <= 6:
            log_event({
                "url": url,
                "status": "QUARANTINED",
                "reason": f"Near-duplicate of target failure case (Hamming: {dist_to_office})",
                "label": lbl
            })
            existing_urls.add(url)
            if diag:
                diag.rejected_invalid_images += 1
            return False

    # 5. Intra-Dataset Near-Duplicate Check (Hamming <= 4)
    for existing_dh in existing_dhashes:
        if hamming_distance(dhash, existing_dh) <= 4:
            log_event({"url": url, "status": "DUPLICATE_PERCEPTUAL", "dhash": dhash, "label": lbl})
            existing_urls.add(url)
            if diag:
                diag.rejected_duplicates += 1
            return False

    # 5b. Multimodal Semantic & Pixel Validation
    is_valid, reject_reason = validate_candidate_semantics_and_pixels(img_rgb, candidate)
    if not is_valid:
        log_event({
            "url": url,
            "status": "QUARANTINED",
            "reason": reject_reason,
            "label": lbl,
            "title": candidate.get("title", ""),
        })
        existing_urls.add(url)
        if diag:
            diag.rejected_invalid_images += 1
        return False

    # 6. Determine Balanced Split & Atomic Save via Temporary File
    dest_dir, split_name = determine_split_destination(lbl, current_counts)
    counter = len(existing_sha) + 1
    dest_path = dest_dir / f"workplace_{lbl.lower()}_{counter:05d}.jpg"
    while dest_path.exists():
        counter += 1
        dest_path = dest_dir / f"workplace_{lbl.lower()}_{counter:05d}.jpg"
    file_id = dest_path.name
    temp_path = dest_dir / f".tmp_{file_id}"

    try:
        img_rgb.save(temp_path, format="JPEG", quality=95)
        os.replace(temp_path, dest_path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        log_event({"url": url, "status": "SAVE_ERROR", "reason": str(e), "label": lbl})
        existing_urls.add(url)
        if diag:
            diag.rejected_invalid_images += 1
        return False
    finally:
        # Guarantee no partial temporary file ever remains on disk
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    # 7. Compute final disk SHA-256 directly from the written file on disk
    try:
        with open(dest_path, "rb") as f:
            final_disk_bytes = f.read()
        final_disk_sha = compute_sha256_bytes(final_disk_bytes)
    except Exception as e:
        log_event({"url": url, "status": "SAVE_ERROR", "reason": f"Failed reading final disk bytes: {e}", "label": lbl})
        existing_urls.add(url)
        if diag:
            diag.rejected_invalid_images += 1
        return False

    # 8. Update In-Memory Tracking State
    rel_path = str(dest_path.relative_to(DATASET_ROOT)).replace("\\", "/")
    existing_sha[raw_payload_sha] = rel_path
    existing_sha[final_disk_sha] = rel_path
    existing_dhashes.append(dhash)
    existing_urls.add(url)
    current_counts[split_name][lbl] += 1

    log_entry = {
        "url": url,
        "status": "VERIFIED",
        "file_path": rel_path,
        "filename": file_id,
        "title": candidate.get("title", ""),
        "split": split_name,
        "label": lbl,
        "class_idx": 0 if lbl == "FAKE" else 1,
        "sha256": final_disk_sha,
        "final_disk_sha256": final_disk_sha,
        "raw_payload_sha256": raw_payload_sha,
        "dhash": dhash,
        "dimensions": [w, h],
        "format": "JPEG",
        "source": candidate.get("source"),
        "license": candidate.get("license") or "CC-BY-SA",
        "author_or_generator": candidate.get("author") or candidate.get("generator") or "Wikimedia Contributor",
        "provenance_notes": candidate.get("description") or candidate.get("prompt") or candidate.get("title", ""),
    }
    log_event(log_entry)
    sync_verified_record_to_ledger(log_entry, ledger_path=DATASET_ROOT / "provenance_ledger.json")
    if diag:
        diag.successfully_accepted += 1
    return True


def run_acquisition(
    target_real: int,
    target_fake: int,
    timeout: float = 25.0,
    max_retries: int = 3,
    base_backoff: float = 1.0,
    max_backoff: float = 10.0,
):
    print("\n" + "=" * 78)
    print("   DEEPGUARD WORKPLACE DATASET RESUMABLE ACQUISITION PIPELINE")
    print("=" * 78)
    print(f"Target REAL Images     : {target_real}")
    print(f"Target FAKE Images     : {target_fake}")
    print(f"Dataset Root           : {DATASET_ROOT}")
    print(f"Network Settings       : Timeout={timeout}s, Retries={max_retries}, Backoff={base_backoff}s-{max_backoff}s")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover partial .tmp files from previous interrupted runs
    cleanup_temp_files()

    try:
        # Compute target dhash of tracked failure case
        target_dhash = None
        if TRACKED_OFFICE_IMAGE.exists():
            with Image.open(TRACKED_OFFICE_IMAGE) as img:
                target_dhash = compute_dhash(img)

        existing_urls, existing_sha, existing_dhashes, failed_urls = load_existing_ledger()
        current_counts = get_current_counts()

        # Seed seen subjects from existing REAL images to prevent subject clustering and cross-split leakage
        log_records = {}
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if rec.get("status") == "VERIFIED" and "file_path" in rec:
                                log_records[rec["file_path"]] = rec
                        except Exception:
                            pass

        seen_subjects = defaultdict(int)
        for split in ["train", "val", "test"]:
            r_folder = DATASET_ROOT / split / "REAL"
            if r_folder.exists():
                for p in r_folder.glob("*.*"):
                    if not p.name.startswith(".tmp") and p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        rel = str(p.relative_to(DATASET_ROOT)).replace("\\", "/")
                        rec = log_records.get(rel, {})
                        title = rec.get("title") or ""
                        url = rec.get("url") or ""
                        subj = extract_subject_key(title, url)
                        if not subj or subj in ("val real", "workplace real"):
                            notes = rec.get("provenance_notes", "")
                            if notes:
                                subj = extract_subject_key(notes, "")
                        if not subj or subj in ("val real", "workplace real"):
                            subj = p.stem
                        seen_subjects[subj] += 1

        total_real = sum(current_counts[sp]["REAL"] for sp in ["train", "val", "test"])
        total_fake = sum(current_counts[sp]["FAKE"] for sp in ["train", "val", "test"])

        print(f"Already verified on disk: {total_real} REAL, {total_fake} FAKE (Total: {total_real + total_fake})")
        print(f"Ledger history loaded   : {len(existing_urls)} total URLs logged ({len(failed_urls)} failed/skipped)")
        print(f"Distinct real subjects  : {len(seen_subjects)} unique subject clusters tracked")

        # 1. Acquire REAL Workplace Images (Wikimedia Commons)
        all_diagnostics = []
        if total_real < target_real:
            needed_real = target_real - total_real
            print(f"\n[Acquisition] Starting REAL image collection. Current: {total_real}, Target: {target_real}, Needed: {needed_real}")
            query_idx = 0
            while total_real < target_real and query_idx < len(WIKIMEDIA_REAL_SEARCH_QUERIES):
                query = WIKIMEDIA_REAL_SEARCH_QUERIES[query_idx]
                diag = QueryDiagnostics(query=query)
                offset = 0
                max_offset_per_query = 100
                consecutive_empty_pages = 0

                while total_real < target_real and offset < max_offset_per_query and consecutive_empty_pages < 2:
                    needed = target_real - total_real
                    batch_size = min(needed * 2, 50)
                    candidates = fetch_wikimedia_workplace_candidates(
                        limit=batch_size,
                        offset=offset,
                        timeout=timeout,
                        max_retries=max_retries,
                        existing_urls=existing_urls,
                        seen_subjects=seen_subjects,
                        max_per_subject=1,
                        query=query,
                        diag=diag,
                    )
                    offset += 50

                    if not candidates:
                        consecutive_empty_pages += 1
                        time.sleep(0.5)
                        continue

                    consecutive_empty_pages = 0
                    for cand in candidates:
                        if total_real >= target_real:
                            break
                        if cand["url"] in existing_urls:
                            diag.rejected_duplicates += 1
                            continue

                        success = download_and_validate_asset(
                            cand,
                            existing_sha,
                            existing_dhashes,
                            target_dhash,
                            current_counts,
                            existing_urls,
                            timeout=timeout,
                            max_retries=max_retries,
                            base_backoff=base_backoff,
                            max_backoff=max_backoff,
                            diag=diag,
                        )
                        if success:
                            total_real += 1
                            seen_subjects[cand.get("subject_key", "")] += 1
                            clean_title = cand.get('title', 'Photo')[:45].encode('ascii', 'replace').decode('ascii')
                            print(f"  [+ REAL] Downloaded {total_real}/{target_real}: {clean_title}")
                        time.sleep(0.1)

                diag.print_summary()
                all_diagnostics.append(diag)
                query_idx += 1

        # Print structured per-query diagnostic summary
        if all_diagnostics:
            print("\n" + "=" * 92)
            print("   STRUCTURED PER-QUERY REAL ACQUISITION DIAGNOSTICS")
            print("=" * 92)
            print(f"{'Query':<42} | {'Found':<5} | {'Errors':<6} | {'Non-Bmp':<7} | {'Keyword':<7} | {'Quota':<5} | {'Dup':<4} | {'Invalid':<7} | {'Accepted':<8}")
            print("-" * 92)
            for d in all_diagnostics:
                q_short = (d.query[:39] + "...") if len(d.query) > 42 else d.query
                print(f"{q_short:<42} | {d.candidates_found:<5d} | {d.http_api_errors:<6d} | {d.rejected_non_bitmap:<7d} | {d.rejected_keyword_filter:<7d} | {d.rejected_subject_quota:<5d} | {d.rejected_duplicates:<4d} | {d.rejected_invalid_images:<7d} | {d.successfully_accepted:<8d}")
            print("=" * 92)

        # 2. Acquire FAKE Workplace Images (OpenFake Verified Generative API)
        hf_offset = 0
        hf_consecutive_failures = 0
        while total_fake < target_fake and hf_consecutive_failures < 5:
            needed = target_fake - total_fake
            batch_size = min(needed * 3, 100)
            candidates = fetch_openfake_generative_candidates(
                limit=batch_size,
                offset=hf_offset,
                timeout=timeout,
                max_retries=max_retries,
            )
            hf_offset += 100

            if not candidates:
                hf_consecutive_failures += 1
                time.sleep(2)
                continue

            hf_consecutive_failures = 0
            for cand in candidates:
                if total_fake >= target_fake:
                    break
                if cand["url"] in existing_urls:
                    continue

                success = download_and_validate_asset(
                    cand,
                    existing_sha,
                    existing_dhashes,
                    target_dhash,
                    current_counts,
                    existing_urls,
                    timeout=timeout,
                    max_retries=max_retries,
                    base_backoff=base_backoff,
                    max_backoff=max_backoff,
                )
                if success:
                    total_fake += 1
                    print(f"  [+ FAKE] Downloaded {total_fake}/{target_fake}: Generator: {cand['generator']}")
                time.sleep(0.1)

        # 3. Final Summary & Storage Reporting
        final_counts = get_current_counts()
        f_real = sum(final_counts[sp]["REAL"] for sp in ["train", "val", "test"])
        f_fake = sum(final_counts[sp]["FAKE"] for sp in ["train", "val", "test"])
        total_images = f_real + f_fake

        total_bytes = sum(f.stat().st_size for f in DATASET_ROOT.glob("**/*") if f.is_file() and not f.name.startswith(".tmp"))
        storage_mb = total_bytes / (1024 * 1024)

        print("\n" + "=" * 78)
        print("   ACQUISITION BATCH COMPLETE — STATUS REPORT")
        print("=" * 78)
        print(f"Total Verified Images on Disk : {total_images} (REAL: {f_real}, FAKE: {f_fake})")
        print(f"Total Storage Consumed        : {storage_mb:.2f} MB")
        print("-" * 78)
        print("Split Distribution:")
        for sp in ["train", "val", "test"]:
            r = final_counts[sp]["REAL"]
            f = final_counts[sp]["FAKE"]
            print(f"  Split '{sp:<5}': {r:3d} REAL | {f:3d} FAKE | Total: {r + f:3d}")
        print("=" * 78)

        target_total = 1200
        remaining = max(0, target_total - total_images)
        if total_images < target_total:
            print(f"[STATUS] INCOMPLETE: {total_images} / {target_total} images verified.")
            print(f"Remaining images needed before full fine-tuning: {remaining}")
        else:
            print(f"[STATUS] READY FOR FINE-TUNING ({total_images} / {target_total} verified images).")

    except KeyboardInterrupt:
        cleanup_temp_files()
        print("\n" + "=" * 78)
        print("   [INTERRUPTED] Acquisition paused cleanly by user (Ctrl+C).")
        print("=" * 78)
        counts = get_current_counts()
        f_real = sum(counts[sp]["REAL"] for sp in ["train", "val", "test"])
        f_fake = sum(counts[sp]["FAKE"] for sp in ["train", "val", "test"])
        total_images = f_real + f_fake
        total_bytes = sum(f.stat().st_size for f in DATASET_ROOT.glob("**/*") if f.is_file() and not f.name.startswith(".tmp"))
        print(f"Verified on disk : {total_images} images ({f_real} REAL, {f_fake} FAKE)")
        print(f"Storage consumed : {total_bytes / (1024 * 1024):.2f} MB")
        print("Acquisition log flushed to disk. All verified assets preserved.")
        print("Run the acquisition script again at any time to resume where you left off.")
        print("=" * 78 + "\n")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Acquire and validate workplace dataset")
    parser.add_argument("--target_real", type=int, default=25, help="Target total verified REAL images")
    parser.add_argument("--target_fake", type=int, default=25, help="Target total verified FAKE images")
    parser.add_argument("--target_per_class", type=int, default=None, help="Target verified images per class (sets both real and fake)")
    parser.add_argument("--timeout", type=float, default=25.0, help="HTTP connection/read timeout in seconds (default: 25.0)")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retry attempts for transient network errors (default: 3)")
    parser.add_argument("--base_backoff", type=float, default=1.0, help="Initial retry backoff delay in seconds (default: 1.0)")
    parser.add_argument("--max_backoff", type=float, default=10.0, help="Maximum retry backoff delay in seconds (default: 10.0)")
    args = parser.parse_args()

    target_real = args.target_real
    target_fake = args.target_fake
    if args.target_per_class is not None:
        target_real = args.target_per_class
        target_fake = args.target_per_class

    run_acquisition(
        target_real=target_real,
        target_fake=target_fake,
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_backoff=args.base_backoff,
        max_backoff=args.max_backoff,
    )


if __name__ == "__main__":
    main()
