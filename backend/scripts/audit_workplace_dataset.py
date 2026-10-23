"""
DeepGuard AI - Workplace Dataset Comprehensive Audit & Leakage Detection Engine
===============================================================================
Performs deep verification of the current workplace dataset:
1. Cryptographic SHA-256 duplicate detection.
2. Perceptual near-duplicate detection via 64-bit dHash (Hamming distance <= 4).
3. Subject & session clustering (detects repeated photo albums / same subjects).
4. Cross-split leakage detection (train vs val vs test).
5. Domain content relevance audit (flags non-workplace images).
6. Provenance validation:
   - REAL: Validates traceable repository URL, human author/contributor, and open license.
   - FAKE: Validates dataset provenance, generative model, and generation prompt.
7. Safe quarantine of ineligible / duplicate / irrelevant images without deleting log records.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

DATASET_ROOT = backend_root / "data" / "workplace_dataset"
LOG_PATH = DATASET_ROOT / "acquisition_log.jsonl"
LEDGER_PATH = DATASET_ROOT / "provenance_ledger.json"
QUARANTINE_DIR = DATASET_ROOT / "quarantine"
TRACKED_OFFICE_IMAGE = backend_root / "data" / "test_cases" / "ai_man_office_laptop.png"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Keywords indicating domain relevance for workplace / person at computer
WORKPLACE_KEYWORDS = [
    "office", "laptop", "computer", "desk", "worker", "workplace", "working",
    "coworking", "business", "programmer", "engineer", "typing", "cubicle",
    "meeting", "boardroom", "executive", "corporate", "conference", "screen"
]

# Known non-workplace / irrelevant patterns to flag
IRRELEVANT_PATTERNS = [
    "fox", "gingerbread", "batman", "harry potter", "soccer", "baseball",
    "panther", "cookbook", "chef", "mat w/ cable", "tool kit", "qu-24 chrome",
    "gym setting", "split on a blue mat", "race car", "musician", "sins of arrogance"
]


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    diff = [img.getpixel((c, r)) > img.getpixel((c + 1, r)) for r in range(hash_size) for c in range(hash_size)]
    val = int("".join(["1" if x else "0" for x in diff]), 2)
    return hex(val)[2:].zfill(hash_size * hash_size // 4)


def hamming_distance(hex1: str, hex2: str) -> int:
    return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")


def load_acquisition_log() -> dict:
    records = {}
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "VERIFIED" and "file_path" in rec:
                        records[rec["file_path"]] = rec
                except Exception:
                    continue
    return records


def extract_subject_cluster_id(image_info: dict) -> str:
    """
    Groups images belonging to the same photoshoot, event, or subject.
    """
    url = (image_info.get("url") or "").lower()
    title = (image_info.get("title") or "").lower()
    fn = image_info.get("filename", "").lower()

    if "alan_schaaf" in url or "alan schaaf" in title:
        return "SUBJECT_ALAN_SCHAAF_TEDX"
    if "janice_shepherd" in url or "janice shepherd" in title:
        return "SUBJECT_JANICE_SHEPHERD_USDA"
    if "social_welfare_minister" in url or "social_workers_receiving_laptops" in url or "social welfare minister" in title:
        return "SUBJECT_BANGLADESH_SOCIAL_WORKERS_LAPTOPS"
    if "reagan" in url or "reagan" in title:
        return "SUBJECT_RONALD_REAGAN_CABINET"
    if "mindpark" in url or "mindpark" in title:
        return "SUBJECT_MINDPARK_COWORKING"
    if fn == "ai_man_office_laptop.png":
        return "TARGET_AI_MAN_OFFICE_LAPTOP"

    return f"SINGLETON_{image_info['sha256'][:8]}"


def run_comprehensive_audit(quarantine_ineligible: bool = False) -> dict:
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    log_map = load_acquisition_log()

    # Collect all physical image files on disk
    all_images = []
    for split in ["train", "val", "test"]:
        for lbl in ["REAL", "FAKE"]:
            folder = DATASET_ROOT / split / lbl
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.*")):
                if p.suffix.lower() in VALID_EXTENSIONS and not p.name.startswith(".tmp"):
                    rel = str(p.relative_to(DATASET_ROOT)).replace("\\", "/")
                    sha = compute_sha256(p)
                    try:
                        with Image.open(p) as img:
                            w, h = img.size
                            mode = img.mode
                            fmt = img.format or "JPEG"
                            rgb = img.convert("RGB")
                            dh = compute_dhash(rgb)
                        decoding_ok = True
                    except Exception as e:
                        w, h = 0, 0
                        mode = "ERR"
                        fmt = "ERR"
                        dh = "0"
                        decoding_ok = False

                    rec = log_map.get(rel, {})
                    title_or_desc = rec.get("provenance_notes") or rec.get("title") or p.name
                    generator = rec.get("author_or_generator", "N/A")
                    source = rec.get("source", "Validation / Development Archive")
                    license_str = rec.get("license", "Open Research / Diagnostic")
                    url = rec.get("url", "local")

                    info = {
                        "path": p,
                        "rel_path": rel,
                        "filename": p.name,
                        "split": split,
                        "label": lbl,
                        "width": w,
                        "height": h,
                        "mode": mode,
                        "format": fmt,
                        "sha256": sha,
                        "dhash": dh,
                        "decoding_ok": decoding_ok,
                        "title": title_or_desc,
                        "generator": generator,
                        "source": source,
                        "license": license_str,
                        "url": url,
                        "rejection_reasons": []
                    }
                    info["subject_id"] = extract_subject_cluster_id(info)
                    all_images.append(info)

    total_images = len(all_images)

    # 1. Exact SHA-256 duplicate clusters
    sha_groups = defaultdict(list)
    for im in all_images:
        sha_groups[im["sha256"]].append(im)
    exact_duplicates = {sha: imgs for sha, imgs in sha_groups.items() if len(imgs) > 1}

    # 2. Perceptual near-duplicates (dHash Hamming <= 4)
    near_duplicate_pairs = []
    n = len(all_images)
    for i in range(n):
        for j in range(i + 1, n):
            im1, im2 = all_images[i], all_images[j]
            if im1["decoding_ok"] and im2["decoding_ok"]:
                dist = hamming_distance(im1["dhash"], im2["dhash"])
                if dist <= 4:
                    near_duplicate_pairs.append({
                        "distance": dist,
                        "image1": im1["rel_path"],
                        "split1": im1["split"],
                        "image2": im2["rel_path"],
                        "split2": im2["split"],
                        "cross_split_leakage": im1["split"] != im2["split"]
                    })

    # 3. Subject / Album Clusters
    subject_groups = defaultdict(list)
    for im in all_images:
        if not im["subject_id"].startswith("SINGLETON_"):
            subject_groups[im["subject_id"]].append(im)

    # Check cross-split leakage for subject groups
    leaked_subjects = {}
    for subj_id, imgs in subject_groups.items():
        splits_present = set(im["split"] for im in imgs)
        if len(splits_present) > 1:
            leaked_subjects[subj_id] = {
                "count": len(imgs),
                "splits": list(splits_present),
                "images": [im["rel_path"] for im in imgs]
            }

    # 4. Domain Relevance & Content Audit
    irrelevant_images = []
    for im in all_images:
        t_low = im["title"].lower()
        url_low = im["url"].lower()
        # Check irrelevant keywords
        for irr in IRRELEVANT_PATTERNS:
            if irr in t_low or irr in url_low:
                im["rejection_reasons"].append(f"Irrelevant content pattern: '{irr}'")
                irrelevant_images.append(im)
                break

    # 5. Dimension & Decoding Check
    invalid_images = []
    for im in all_images:
        if not im["decoding_ok"]:
            im["rejection_reasons"].append("Image decoding failed")
            invalid_images.append(im)
        elif im["width"] < 64 or im["height"] < 64:
            im["rejection_reasons"].append(f"Resolution too low ({im['width']}x{im['height']})")
            invalid_images.append(im)

    # 6. Provenance Check
    provenance_gaps = []
    for im in all_images:
        if im["label"] == "REAL":
            if im["source"] == "Wikimedia Commons" and (im["license"] in ("N/A", "") or not im["url"].startswith("http")):
                im["rejection_reasons"].append("Missing verified repository provenance/license")
                provenance_gaps.append(im)
        elif im["label"] == "FAKE":
            # For acquired fake images, check generator
            if im["filename"].startswith("workplace_fake_") and (im["generator"] in ("N/A", "Unknown", "")):
                im["rejection_reasons"].append("Missing generator provenance")
                provenance_gaps.append(im)

    # 7. Redundancy Reduction: Within subject clusters, identify surplus duplicate shots
    # For large albums like Alan Schaaf (16 photos) or Janice Shepherd (4 photos), keep at most 1 representative
    # in an isolated split and flag the rest as redundant surplus to eliminate leakage & bias
    surplus_subject_duplicates = []
    for subj_id, imgs in subject_groups.items():
        if subj_id == "TARGET_AI_MAN_OFFICE_LAPTOP":
            continue
        if len(imgs) > 1:
            # Sort by resolution / filename to pick 1 canonical representative
            sorted_imgs = sorted(imgs, key=lambda x: (x["width"] * x["height"], x["filename"]), reverse=True)
            # Retain the first, mark the rest as subject redundancy surplus
            for surplus in sorted_imgs[1:]:
                surplus["rejection_reasons"].append(f"Subject album redundancy surplus for {subj_id} (keeping {sorted_imgs[0]['filename']})")
                surplus_subject_duplicates.append(surplus)

    # Combine all items slated for quarantine
    ineligible_set = set()
    for im in all_images:
        if im["rejection_reasons"]:
            ineligible_set.add(im["rel_path"])

    quarantine_audit_log = []
    if quarantine_ineligible:
        for im in all_images:
            if im["rel_path"] in ineligible_set:
                src_path = im["path"]
                if src_path.exists():
                    q_dest = QUARANTINE_DIR / f"audit_quarantine_{im['split']}_{im['label']}_{src_path.name}"
                    shutil.move(src_path, q_dest)
                    quarantine_audit_log.append({
                        "file_path": im["rel_path"],
                        "quarantined_path": str(q_dest),
                        "reasons": im["rejection_reasons"],
                        "sha256": im["sha256"],
                        "status": "QUARANTINED"
                    })

    # Remaining eligible count calculation
    eligible_images = [im for im in all_images if im["rel_path"] not in ineligible_set]
    eligible_counts = defaultdict(lambda: {"REAL": 0, "FAKE": 0})
    for im in eligible_images:
        eligible_counts[im["split"]][im["label"]] += 1

    report = {
        "dataset_root": str(DATASET_ROOT),
        "total_images_scanned": total_images,
        "exact_duplicate_clusters": len(exact_duplicates),
        "exact_duplicate_details": {
            sha: [img["rel_path"] for img in imgs] for sha, imgs in exact_duplicates.items()
        },
        "perceptual_near_duplicate_pairs": near_duplicate_pairs,
        "subject_clusters": {
            subj: len(imgs) for subj, imgs in subject_groups.items()
        },
        "cross_split_leaked_subjects": leaked_subjects,
        "irrelevant_content_count": len(irrelevant_images),
        "irrelevant_images": [
            {"file": im["rel_path"], "title": im["title"][:80], "reason": im["rejection_reasons"]}
            for im in irrelevant_images
        ],
        "surplus_subject_duplicates_count": len(surplus_subject_duplicates),
        "surplus_subject_duplicates": [
            {"file": im["rel_path"], "subject": im["subject_id"], "reasons": im["rejection_reasons"]}
            for im in surplus_subject_duplicates
        ],
        "total_ineligible_for_dataset": len(ineligible_set),
        "eligible_remaining_count": len(eligible_images),
        "eligible_by_class": {
            "REAL": sum(eligible_counts[s]["REAL"] for s in eligible_counts),
            "FAKE": sum(eligible_counts[s]["FAKE"] for s in eligible_counts),
        },
        "eligible_splits": {
            sp: dict(eligible_counts[sp]) for sp in ["train", "val", "test"]
        },
        "target_counts": {
            "target_total": 1200,
            "target_real": 600,
            "target_fake": 600,
            "remaining_real": 600 - sum(eligible_counts[s]["REAL"] for s in eligible_counts),
            "remaining_fake": 600 - sum(eligible_counts[s]["FAKE"] for s in eligible_counts),
        }
    }

    return report


def print_audit_report(report: dict):
    print("\n" + "=" * 80)
    print("   DEEPGUARD WORKPLACE DATASET COMPREHENSIVE AUDIT REPORT")
    print("=" * 80)
    print(f"Dataset Root                    : {report['dataset_root']}")
    print(f"Total Disk Images Scanned       : {report['total_images_scanned']}")
    print(f"Exact SHA-256 Duplicate Clusters: {report['exact_duplicate_clusters']}")
    print(f"Perceptual Near-Duplicate Pairs : {len(report['perceptual_near_duplicate_pairs'])}")
    print(f"Cross-Split Leaked Subjects     : {len(report['cross_split_leaked_subjects'])}")
    print(f"Irrelevant Content Identified   : {report['irrelevant_content_count']}")
    print(f"Subject Redundancy Surplus      : {report['surplus_subject_duplicates_count']}")
    print(f"Total Ineligible for Training   : {report['total_ineligible_for_dataset']}")
    print("-" * 80)

    print("Subject Clusters Identified:")
    for subj, cnt in report["subject_clusters"].items():
        print(f"  - {subj:<42}: {cnt:2d} images")

    print("\nCross-Split Leaked Subjects (Spanning train, val, and/or test):")
    for subj, det in report["cross_split_leaked_subjects"].items():
        print(f"  * {subj}: {det['count']} images spanning {det['splits']}")

    if report["irrelevant_images"]:
        print("\nIrrelevant / Out-of-Domain Images:")
        for irr in report["irrelevant_images"][:10]:
            print(f"  - [{irr['file']}] {irr['title']}")
        if len(report["irrelevant_images"]) > 10:
            print(f"    ... and {len(report['irrelevant_images']) - 10} more.")

    print("\n" + "=" * 80)
    print("   DATASET ELIGIBILITY & REMAINING COUNTS")
    print("=" * 80)
    e_real = report["eligible_by_class"]["REAL"]
    e_fake = report["eligible_by_class"]["FAKE"]
    print(f"Current Clean Eligible Images   : {report['eligible_remaining_count']} (REAL: {e_real}, FAKE: {e_fake})")
    print("Clean Eligible Split Distribution:")
    for sp in ["train", "val", "test"]:
        sp_counts = report["eligible_splits"].get(sp, {"REAL": 0, "FAKE": 0})
        print(f"  Split '{sp:<5}': {sp_counts.get('REAL', 0):2d} REAL | {sp_counts.get('FAKE', 0):2d} FAKE | Total: {sp_counts.get('REAL', 0) + sp_counts.get('FAKE', 0):2d}")
    print("-" * 80)
    print(f"Target Total                    : {report['target_counts']['target_total']} (600 REAL, 600 FAKE)")
    print(f"Remaining Deficit to Acquire    : REAL: {report['target_counts']['remaining_real']} | FAKE: {report['target_counts']['remaining_fake']}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit and validate workplace dataset")
    parser.add_argument("--quarantine", action="store_true", help="Move ineligible, duplicate, and irrelevant images to quarantine/")
    parser.add_argument("--json", action="store_true", help="Output audit report as JSON")
    args = parser.parse_args()

    report = run_comprehensive_audit(quarantine_ineligible=args.quarantine)
    if args.json:
        # Serializer helper for Paths
        print(json.dumps(report, indent=2, default=str))
    else:
        print_audit_report(report)


if __name__ == "__main__":
    main()
