"""
DeepGuard AI - Strict Semantic Quality Audit & Contact Sheet Generator
======================================================================
Inspects image pixels, evaluates semantic domain relevance, generates visual contact sheets,
and identifies confirmed irrelevant vs ambiguous images across the workplace dataset.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torchvision.models as models

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_ROOT = backend_root / "data" / "workplace_dataset"
LOG_PATH = DATASET_ROOT / "acquisition_log.jsonl"
LEDGER_PATH = DATASET_ROOT / "provenance_ledger.json"
QUARANTINE_DIR = DATASET_ROOT / "quarantine"
CONTACT_SHEETS_DIR = DATASET_ROOT / "contact_sheets"
ARTIFACTS_DIR = Path(r"C:\Users\admin\.gemini\antigravity-ide\brain\01a982e7-b34f-48a5-9c09-d0092ed43ca3")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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

def load_provenance_records() -> dict:
    records = {}
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("status") == "VERIFIED" and "file_path" in rec:
                            records[rec["file_path"]] = rec
                    except Exception:
                        pass
    return records

class VisionInspector:
    def __init__(self):
        print("Loading ImageNet vision model & face detectors...")
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.eval()
        self.categories = weights.meta["categories"]
        self.transform = weights.transforms()

        # OpenCV Haar cascade face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

    def analyze_image(self, img_pil: Image.Image) -> dict:
        rgb_img = img_pil.convert("RGB")
        w, h = rgb_img.size

        # 1. ResNet50 Predictions
        tensor = self.transform(rgb_img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)

        predictions = [
            (self.categories[idx.item()], float(prob.item()))
            for prob, idx in zip(top5_prob, top5_idx)
        ]

        # 2. Fast Face Detection and Pixel Metrics on downscaled copy
        max_dim = 640
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            small_w, small_h = int(w * scale), int(h * scale)
            small_rgb = rgb_img.resize((small_w, small_h), Image.Resampling.BILINEAR)
            np_img = np.array(small_rgb)
        else:
            np_img = np.array(rgb_img)

        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        num_faces = len(faces)
        if num_faces == 0:
            profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
            num_faces = len(profiles)

        # 3. Visual Pixel Properties
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        val_mean = float(np.mean(hsv[:, :, 2]))

        # Edge density (detect line drawings, text, book scans)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.mean(edges > 0))

        return {
            "width": w,
            "height": h,
            "num_faces": num_faces,
            "top_prediction": predictions[0][0],
            "top_prob": predictions[0][1],
            "top5": predictions,
            "sat_mean": sat_mean,
            "val_mean": val_mean,
            "edge_density": edge_density,
        }


# Domain evaluation rules
CONFIRMED_IRRELEVANT_PATTERNS = [
    # Historical hardware / artifacts without human workplace
    ("cromemco", "Antique computer hardware unit without people/workplace scene"),
    ("crashed computer", "Crashed hardware / damaged equipment without workplace context"),
    ("crashed_computer", "Crashed hardware / damaged equipment without workplace context"),
    ("bioart", "3D scientific icon / rendering without human workplace context"),
    # Art statues / sculptures / drawings
    ("faun", "Museum sculpture / art statue (Satyrkopf Faun)"),
    ("satyrkopf", "Museum sculpture / art statue (Satyrkopf Faun)"),
    ("la bella principessa", "16th-century Renaissance chalk/ink portrait on vellum"),
    ("dotkamina", "Artistic manga / cartoon sketch"),
    # Leisure / vehicles
    ("camper van", "Leisure camper van / RV roadside camping outside workplace domain"),
    ("campervan", "Leisure camper van / RV roadside camping outside workplace domain"),
    ("motor home", "Leisure motor home roadside camping outside workplace domain"),
    # Child / primary school
    ("child remote schooling", "Child primary remote schooling outside workplace domain"),
    ("kids learning how to use", "Schoolchildren classroom exercise outside adult workplace domain"),
    ("child in medical consultation", "Pediatric consultation outside workplace domain"),
    ("form and function (2475731)", "Art exhibition / architectural form outside workplace domain"),
    # Exact target failures from user request
    ("le tour du monde", "19th-century travel periodical woodcut / book crop (Le Tour du monde)"),
    ("denier vulgate", "Antique numismatic coin / biblical text illustration (Denier Vulgate)"),
    ("denier d’auguste", "Antique numismatic coin / medal (Denier d’Auguste)"),
    ("denier d'auguste", "Antique numismatic coin / medal (Denier d’Auguste)"),
    ("grand bronze vulgate", "Antique bronze coin / medallion (Grand bronze Vulgate)"),
    ("moyen bronze vulgate", "Antique bronze coin / medallion (Moyen bronze Vulgate)"),
    ("petit bronze vulgate", "Antique bronze coin / medallion (Petit bronze Vulgate)"),
    ("marché égyptien vulgate", "Antique marketplace illustration from 19th-century Vulgate"),
    ("marche egyptien vulgate", "Antique marketplace illustration from 19th-century Vulgate"),
    ("la plus petite monnaie de bronze", "Antique bronze coin illustration (Vulgate Glaire)"),
    ("mine d’antiochus", "Antique numismatic illustration (Mine d'Antiochus)"),
    ("mine dantiochus", "Antique numismatic illustration (Mine d'Antiochus)"),
    ("musiciens de suse", "Antique bas-relief engraving (Musiciens de Suse Vulgate)"),
    ("flag of beverly hills", "Municipal flag / graphic emblem (Beverly Hills)"),
    ("flag of west hollywood", "Municipal flag / graphic emblem (West Hollywood)"),
    ("warshipswidth", "Statistical P-P / Q-Q plot diagram (Warships width)"),
    ("warships width", "Statistical P-P / Q-Q plot diagram (Warships width)"),
    ("feeding a laboratory rat", "Laboratory animal / rodent experiment (Feeding a laboratory rat)"),
    ("mount kilimanjaro", "Aerial mountain landscape / orbital earth observation (Mount Kilimanjaro)"),
    ("watercolor depiction of a wikipedia user", "Artistic watercolor illustration / drawing (Wiki Maddie)"),
    # Additional plots / diagrams / algorithmic graphs
    ("operations.jpg", "Algorithmic graph / HITS authority update formula diagram"),
    ("x-y plot of algorithmically-generated ai art", "Statistical prompt grid plot of generative art"),
    ("rayleigh-taylor instability", "Computational fluid dynamics simulation visualization"),
    ("geological features of the country west of the mississippi", "1857 antique geological map"),
    ("hauptbahnhof-karlsruhe", "Outdoor public railway transit architecture"),
    ("virtual fencing project at fd mcmaster laboratory", "Outdoor cattle / livestock virtual fencing field experiment"),
    # Scanned 17th/18th/19th-century book crops & manuscripts
    ("boussinesq", "1901 scanned physics treatise page crop (Boussinesq)"),
    ("blondel - l'art de jetter les bombes", "1683 military treatise illustration (Blondel)"),
    ("buddhist wheel of law", "Scanned book illustration (History of India Vol 2)"),
    ("contes des fées", "1872 fairy tales book page crop (Perrault)"),
    ("correspondance de guillaume le taciturne", "1857 historical correspondence book crop"),
    ("gemelli careri", "1704 travel publication page crop (Gemelli Careri)"),
    ("gozzi - le fiabe", "1884 theatre book page crop (Gozzi)"),
    ("journal de psychologie", "1904 medical journal page crop"),
    ("la grivoise du temps", "1747 handwritten manuscript page crop"),
    ("le magasin des enfants", "1843 children's book page crop"),
    ("magasin pittoresque", "19th-century magazine engraving crop"),
    ("हिंदी विश्वकोष", "Scanned encyclopedia book page crop"),
    ("কাশীদাসী মহাভারত", "Scanned Bengali Mahabharata book page crop"),
    ("le cabinet des fées", "Scanned fairy tales collection page crop"),
    ("scouffi - au poiss", "1929 novel publication page crop (Scouffi)"),
    ("a time travel dialogue", "Scanned book page crop"),
    ("eiffel - la tour eiffel en 1900", "1902 engineering book page crop (Eiffel)"),
    ("arago - souvenirs", "1840 memoirs page crop (Arago)"),
    ("démosthène - œuvres complètes", "1819 classical treatise page crop (Démosthène)"),
    ("demosthene - oeuvres completes", "1819 classical treatise page crop (Démosthène)"),
    ("il libro dei re", "1887 Persian epic translation page crop"),
    ("recueil des historiens", "Scanned historical annals page crop"),
    ("cabanès - grands névropathes", "1931 medical biography page crop (Cabanès)"),
    ("sept lettres de mérimée", "1898 literary letters page crop (Mérimée)"),
    ("la corée libre", "1920 political bulletin page crop"),
    # AI art incorrectly labeled as REAL
    ("ai end-scenario 2", "Synthetic generative AI artwork in REAL class"),
    ("loss, in first-person view", "Collection of generative AI artwork in REAL class"),
    # Non-workplace generic landscapes
    ("local_real_landscape", "Generic outdoor nature landscape"),
    ("val_real_03.jpg", "Generic alpine mountain landscape"),
]

AMBIGUOUS_PATTERNS = [
    ("telework thailand", "Digital nomad leisure / beach setting with laptop"),
    ("pause by exey panteleev", "Artistic break portrait with subtle computer cue"),
    ("margaret hamilton", "Historic portrait of software pioneer next to code printouts"),
    ("digital nomad remote work thailand", "Digital nomad leisure / beach setting with laptop"),
    ("man typing on a laptop working as a digital nomad", "Digital nomad mobile van interior"),
    ("michael holley computer 1978", "Vintage personal computer workstation without human"),
    ("edsac (13)", "Historic mainframe computing hardware"),
    ("regular life of laboratory of computer security problems", "Casual pose in computer security office"),
]

METADATA_REJECT_REGEX_AUDIT = [
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


def classify_semantic_relevance(image_info: dict, vision_data: dict) -> tuple[str, str]:
    """
    Evaluates both visual pixel predictions and metadata to categorize images into:
    - VALID: Confirmed authentic/synthetic workplace scene
    - CONFIRMED_IRRELEVANT: Clearly out of domain
    - AMBIGUOUS: Borderline case requiring manual review
    """
    lbl = image_info["label"]
    title = (image_info.get("title") or "").lower()
    fn = image_info["filename"].lower()
    notes = (image_info.get("provenance_notes") or "").lower()
    top_pred = vision_data["top_prediction"].lower()
    top5_str = " ".join(p[0].lower() for p in vision_data["top5"])

    combined_text = f"{title} {fn} {notes} {top5_str}"

    # 1. Check confirmed irrelevant patterns
    for pat, reason in CONFIRMED_IRRELEVANT_PATTERNS:
        if pat in combined_text or pat in title:
            return "CONFIRMED_IRRELEVANT", reason

    # 2. Check metadata regex patterns for REAL images
    if lbl == "REAL":
        comb_meta = f"{title} {fn} {notes}"
        for pat, reason in METADATA_REJECT_REGEX_AUDIT:
            if pat.search(comb_meta):
                return "CONFIRMED_IRRELEVANT", reason

    # 3. Visual Pixel Flags for Non-Workplace Objects
    # Scanned book / document / pure text page
    if any(k in top5_str for k in ["book jacket", "comic book", "menu", "packet", "envelope", "web site"]):
        if vision_data["sat_mean"] < 18 and vision_data["edge_density"] > 0.075 and vision_data["num_faces"] == 0:
            return "CONFIRMED_IRRELEVANT", f"Scanned book page or printed document ({top_pred})"

    # Statistical plot / diagram detected in pixels
    if any(k in top_pred for k in ["web site", "envelope", "menu"]) and vision_data["sat_mean"] < 5.0 and vision_data["edge_density"] > 0.030:
        return "CONFIRMED_IRRELEVANT", f"Statistical chart / plot diagram ({top_pred})"

    # Flat flag / graphic emblem in pixels (must have no people and high saturation)
    if (("flagpole" in top_pred or "puck" in top_pred) and vision_data["num_faces"] == 0 and vision_data["sat_mean"] > 120 and lbl == "REAL"):
        return "CONFIRMED_IRRELEVANT", f"Municipal flag / graphic emblem ({top_pred})"

    # Vehicles / RVs / Camper vans detected in pixels
    if any(v in top_pred for v in ["moving van", "recreational vehicle", "minibus", "police van", "trailer truck"]):
        if "camper" in combined_text or "van" in combined_text:
            return "CONFIRMED_IRRELEVANT", f"Outdoor camper van / vehicle scene ({top_pred})"

    # Statues / sculptures / art pieces
    if any(s in top_pred for s in ["sculpture", "statue", "bronze", "brass", "monument"]):
        return "CONFIRMED_IRRELEVANT", f"Museum sculpture / statue ({top_pred})"

    # Mountain / landscape detected in pixels
    if any(m in top_pred for m in ["volcano", "alp", "cliff", "promontory", "valley"]):
        if vision_data["num_faces"] == 0 and lbl == "REAL":
            return "CONFIRMED_IRRELEVANT", f"Outdoor natural landscape / geology ({top_pred})"

    # Animals / rodents detected in pixels (without workplace context)
    has_office_context = any(k in f"{title} {notes}" for k in ["computer", "laptop", "office", "desk", "workstation", "home office", "studying"])
    if not has_office_context and any(a in top_pred for a in ["hamster", "guinea pig", "rat", "llama"]):
        if vision_data["num_faces"] <= 1 and lbl == "REAL":
            return "CONFIRMED_IRRELEVANT", f"Animal subject ({top_pred})"

    # 4. For FAKE images:
    if lbl == "FAKE":
        if any(irr in notes for irr in ["fox", "gingerbread", "batman", "baseball", "soccer", "recipe"]):
            return "CONFIRMED_IRRELEVANT", f"Irrelevant synthetic prompt: {notes[:60]}"

    # 5. Check Ambiguous Cases
    for pat, reason in AMBIGUOUS_PATTERNS:
        if pat in combined_text:
            return "AMBIGUOUS", reason

    # If no face detected and top predictions contain no computer/office cues
    workplace_cues = ["laptop", "notebook", "desk", "computer", "keyboard", "monitor", "screen", "mouse"]
    has_computer_cue = any(c in top5_str for c in workplace_cues)
    if not has_computer_cue and vision_data["num_faces"] == 0 and lbl == "REAL":
        if any(c in title for c in ["office", "desk", "workstation", "room"]):
            return "AMBIGUOUS", f"Empty room / setting without visible computer or person ({top_pred})"

    return "VALID", "Verified workplace scene with computer/desk/worker"


def generate_contact_sheet(
    image_entries: list,
    output_path: Path,
    title_text: str,
    cols: int = 8,
    thumb_size: int = 160,
    banner_height: int = 40,
):
    """
    Renders a structured visual contact sheet with color-coded evaluation borders:
    - Green: VALID
    - Red: CONFIRMED_IRRELEVANT
    - Orange: AMBIGUOUS
    """
    n = len(image_entries)
    if n == 0:
        return
    rows = (n + cols - 1) // cols

    cell_w = thumb_size + 12
    cell_h = thumb_size + banner_height + 12

    header_h = 50
    sheet_w = cols * cell_w + 20
    sheet_h = header_h + rows * cell_h + 20

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(24, 26, 32))
    draw = ImageDraw.Draw(sheet)

    # Header
    draw.rectangle([(0, 0), (sheet_w, header_h)], fill=(15, 17, 23))
    draw.text((20, 15), title_text, fill=(240, 240, 245))

    status_colors = {
        "VALID": (34, 197, 94),             # Green
        "CONFIRMED_IRRELEVANT": (239, 68, 68), # Red
        "AMBIGUOUS": (245, 158, 11),         # Orange / Amber
    }

    for idx, entry in enumerate(image_entries):
        r = idx // cols
        c = idx % cols
        x = 10 + c * cell_w + 6
        y = header_h + 10 + r * cell_h + 6

        status = entry["semantic_status"]
        border_col = status_colors.get(status, (150, 150, 150))

        # Thumbnail
        try:
            with Image.open(entry["path"]) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                # Pad to square
                tw, th = thumb.size
                tx = x + (thumb_size - tw) // 2
                ty = y + (thumb_size - th) // 2
                sheet.paste(thumb, (tx, ty))
        except Exception:
            draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], fill=(50, 20, 20))
            draw.text((x + 10, y + 50), "DECODE ERROR", fill=(255, 100, 100))

        # Border
        draw.rectangle(
            [(x - 2, y - 2), (x + thumb_size + 1, y + thumb_size + 1)],
            outline=border_col,
            width=3 if status != "VALID" else 1,
        )

        # Label Banner
        banner_y = y + thumb_size + 4
        draw.rectangle([(x, banner_y), (x + thumb_size, banner_y + banner_height - 2)], fill=(32, 35, 42))

        fn_short = entry["filename"]
        if len(fn_short) > 22:
            fn_short = fn_short[:20] + ".."
        draw.text((x + 3, banner_y + 2), fn_short, fill=(220, 220, 230))

        pred_short = entry["top_prediction"][:18]
        status_abbr = "OK" if status == "VALID" else ("REJECT" if status == "CONFIRMED_IRRELEVANT" else "AMBIG")
        draw.text((x + 3, banner_y + 16), f"[{status_abbr}] {pred_short}", fill=border_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    print(f"Saved contact sheet: {output_path} ({n} images)")


def run_strict_semantic_audit(quarantine: bool = False) -> dict:
    CONTACT_SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    provenance_map = load_provenance_records()
    inspector = VisionInspector()

    print("\n" + "=" * 80)
    print("   DEEPGUARD WORKPLACE DATASET STRICT SEMANTIC QUALITY AUDIT")
    print("=" * 80)

    all_entries = []
    by_group = defaultdict(list)

    for split in ["train", "val", "test"]:
        for lbl in ["REAL", "FAKE"]:
            folder = DATASET_ROOT / split / lbl
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*.*")):
                if p.suffix.lower() not in VALID_EXTENSIONS or p.name.startswith(".tmp"):
                    continue

                rel = str(p.relative_to(DATASET_ROOT)).replace("\\", "/")
                rec = provenance_map.get(rel, {})

                sha = compute_sha256(p)
                try:
                    with Image.open(p) as img_pil:
                        dh = compute_dhash(img_pil)
                        vision_data = inspector.analyze_image(img_pil)
                except Exception as e:
                    dh = "0"
                    vision_data = {
                        "width": 0, "height": 0, "num_faces": 0,
                        "top_prediction": "ERROR", "top_prob": 0.0,
                        "top5": [], "sat_mean": 0, "val_mean": 0, "edge_density": 0
                    }

                title = rec.get("title") or rec.get("provenance_notes") or p.name
                info = {
                    "path": p,
                    "rel_path": rel,
                    "filename": p.name,
                    "split": split,
                    "label": lbl,
                    "sha256": sha,
                    "dhash": dh,
                    "title": title,
                    "license": rec.get("license", "N/A"),
                    "author": rec.get("author_or_generator", "N/A"),
                    "url": rec.get("url", "local"),
                    "provenance_notes": rec.get("provenance_notes", ""),
                    "width": vision_data["width"],
                    "height": vision_data["height"],
                    "num_faces": vision_data["num_faces"],
                    "top_prediction": vision_data["top_prediction"],
                    "top_prob": vision_data["top_prob"],
                    "edge_density": vision_data["edge_density"],
                }

                status, reason = classify_semantic_relevance(info, vision_data)
                info["semantic_status"] = status
                info["semantic_reason"] = reason

                all_entries.append(info)
                by_group[f"{split}_{lbl}"].append(info)
                if len(all_entries) % 50 == 0:
                    print(f"  Analyzed {len(all_entries)}/400 images...")

    # 1. Count findings
    confirmed_irrelevant = [e for e in all_entries if e["semantic_status"] == "CONFIRMED_IRRELEVANT"]
    ambiguous = [e for e in all_entries if e["semantic_status"] == "AMBIGUOUS"]
    valid = [e for e in all_entries if e["semantic_status"] == "VALID"]

    # 2. Contact Sheets Generation (Including dedicated Quarantine and Ambiguous contact sheets)
    print("\nGenerating Contact Sheets for human visual review...")
    contact_sheet_paths = []

    # Dedicated Quarantine Contact Sheet
    if confirmed_irrelevant:
        cs_quarantine_path = CONTACT_SHEETS_DIR / "contact_sheet_quarantined.jpg"
        generate_contact_sheet(
            confirmed_irrelevant,
            cs_quarantine_path,
            title_text=f"DeepGuard Audit - Confirmed Irrelevant Quarantine Candidates ({len(confirmed_irrelevant)} images)",
            cols=10 if len(confirmed_irrelevant) > 50 else 8,
        )
        contact_sheet_paths.append(cs_quarantine_path)
        if ARTIFACTS_DIR.exists():
            shutil.copy2(cs_quarantine_path, ARTIFACTS_DIR / "contact_sheet_quarantined.jpg")

    # Dedicated Ambiguous Contact Sheet
    if ambiguous:
        cs_ambiguous_path = CONTACT_SHEETS_DIR / "contact_sheet_ambiguous.jpg"
        generate_contact_sheet(
            ambiguous,
            cs_ambiguous_path,
            title_text=f"DeepGuard Audit - Ambiguous Workplace Domain Candidates ({len(ambiguous)} images)",
            cols=8,
        )
        contact_sheet_paths.append(cs_ambiguous_path)
        if ARTIFACTS_DIR.exists():
            shutil.copy2(cs_ambiguous_path, ARTIFACTS_DIR / "contact_sheet_ambiguous.jpg")

    # Split / Class Contact Sheets
    for grp_name, entries in by_group.items():
        cs_filename = f"contact_sheet_{grp_name}.jpg"
        cs_path = CONTACT_SHEETS_DIR / cs_filename
        generate_contact_sheet(
            entries,
            cs_path,
            title_text=f"DeepGuard Dataset Review - Split: {grp_name.replace('_', ' ')} ({len(entries)} images)",
            cols=10 if len(entries) > 50 else 8,
        )
        contact_sheet_paths.append(cs_path)

        # Also copy to artifacts dir
        if ARTIFACTS_DIR.exists():
            shutil.copy2(cs_path, ARTIFACTS_DIR / cs_filename)

    # 3. Cryptographic and Perceptual Duplicate checks
    sha_map = defaultdict(list)
    for e in all_entries:
        sha_map[e["sha256"]].append(e["rel_path"])
    exact_duplicates = {sha: paths for sha, paths in sha_map.items() if len(paths) > 1}

    near_duplicates = []
    m = len(all_entries)
    for i in range(m):
        for j in range(i + 1, m):
            e1, e2 = all_entries[i], all_entries[j]
            dist = hamming_distance(e1["dhash"], e2["dhash"])
            if dist <= 4:
                near_duplicates.append({
                    "image1": e1["rel_path"],
                    "image2": e2["rel_path"],
                    "dist": dist,
                    "cross_split": e1["split"] != e2["split"],
                })

    # 4. Quarantine Action if requested
    quarantined_files = []
    if quarantine:
        print("\nExecuting Quarantine for Confirmed Ineligible Images...")
        now_ts = datetime.now(timezone.utc).isoformat()
        for item in confirmed_irrelevant:
            src = item["path"]
            if src.exists():
                dest_name = f"semantic_quarantine_{item['split']}_{item['label']}_{src.name}"
                dest = QUARANTINE_DIR / dest_name
                shutil.move(src, dest)
                quarantined_files.append(item["rel_path"])

                # Record in acquisition_log.jsonl
                log_entry = {
                    "timestamp": now_ts,
                    "url": item["url"],
                    "status": "QUARANTINED",
                    "file_path": item["rel_path"],
                    "quarantined_path": str(dest),
                    "reason": item["semantic_reason"],
                    "label": item["label"],
                    "sha256": item["sha256"],
                    "title": item["title"],
                }
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

        # Update provenance_ledger.json
        if LEDGER_PATH.exists():
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    ledger_data = json.load(f)
            except Exception:
                ledger_data = {"dataset_root": str(DATASET_ROOT), "quarantine_count": 0, "verified_records": [], "quarantined_records": []}
        else:
            ledger_data = {"dataset_root": str(DATASET_ROOT), "quarantine_count": 0, "verified_records": [], "quarantined_records": []}

        quarantined_rel_set = set(quarantined_files)
        old_verified = ledger_data.get("verified_records", [])
        new_verified = [r for r in old_verified if r.get("file_path") not in quarantined_rel_set]

        # Ensure ai_man_office_laptop is annotated properly in verified records if present
        for r in new_verified:
            if "ai_man_office_laptop" in str(r.get("file_path", "")):
                r["provenance_notes"] = "Known development regression benchmark asset. Strictly isolated from training and validation splits. Not an untouched final-test held-out example."

        new_quarantined = ledger_data.get("quarantined_records", [])
        for item in confirmed_irrelevant:
            new_quarantined.append({
                "file_path": item["rel_path"],
                "filename": item["filename"],
                "split": item["split"],
                "label": item["label"],
                "title": item["title"],
                "url": item["url"],
                "reason": item["semantic_reason"],
                "sha256": item["sha256"],
                "timestamp": now_ts,
            })

        ledger_data["verified_records"] = new_verified
        ledger_data["quarantined_records"] = new_quarantined
        ledger_data["quarantine_count"] = len(new_quarantined)

        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger_data, f, indent=2)

    # Remaining counts
    remaining_entries = [e for e in all_entries if e not in confirmed_irrelevant] if quarantine else all_entries
    rem_counts = defaultdict(lambda: {"REAL": 0, "FAKE": 0})
    for e in remaining_entries:
        rem_counts[e["split"]][e["label"]] += 1

    report = {
        "total_scanned": len(all_entries),
        "valid_count": len(valid),
        "confirmed_irrelevant_count": len(confirmed_irrelevant),
        "ambiguous_count": len(ambiguous),
        "confirmed_irrelevant_items": [
            {"file": e["rel_path"], "reason": e["semantic_reason"], "title": e["title"][:75]}
            for e in confirmed_irrelevant
        ],
        "ambiguous_items": [
            {"file": e["rel_path"], "reason": e["semantic_reason"], "title": e["title"][:75]}
            for e in ambiguous
        ],
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": near_duplicates,
        "clean_counts_after": {
            "REAL": sum(rem_counts[s]["REAL"] for s in ["train", "val", "test"]),
            "FAKE": sum(rem_counts[s]["FAKE"] for s in ["train", "val", "test"]),
            "splits": {s: dict(rem_counts[s]) for s in ["train", "val", "test"]},
        },
        "contact_sheets": [str(p) for p in contact_sheet_paths],
    }

    return report


def print_audit_summary(report: dict, quarantine: bool):
    print("\n" + "=" * 80)
    print("   STRICT SEMANTIC QUALITY AUDIT RESULTS")
    print("=" * 80)
    print(f"Total Images Scanned            : {report['total_scanned']}")
    print(f"Confirmed Valid Workplace Scenes: {report['valid_count']}")
    print(f"Confirmed Irrelevant Out-of-Dom : {report['confirmed_irrelevant_count']}")
    print(f"Ambiguous Cases (Review Needed) : {report['ambiguous_count']}")
    print(f"Exact Cryptographic Duplicates  : {len(report['exact_duplicates'])}")
    print(f"Perceptual Near-Duplicates (<=4): {len(report['near_duplicate_pairs'])}")
    print("-" * 80)

    if report["confirmed_irrelevant_items"]:
        print("\nConfirmed Irrelevant Images Flagged for Quarantine:")
        for it in report["confirmed_irrelevant_items"]:
            print(f"  [X] {it['file']:<35} : {it['reason']} | '{it['title']}'")

    if report["ambiguous_items"]:
        print("\nAmbiguous / Borderline Images (Preserved for Human Review):")
        for it in report["ambiguous_items"]:
            print(f"  [?] {it['file']:<35} : {it['reason']} | '{it['title']}'")

    print("\n" + "=" * 80)
    print("   DATASET SPLIT DISTRIBUTION & DEFICIT (Target: 600 REAL / 600 FAKE)")
    print("=" * 80)
    cc = report["clean_counts_after"]
    print(f"Current Clean Images            : Total: {cc['REAL'] + cc['FAKE']} (REAL: {cc['REAL']}, FAKE: {cc['FAKE']})")
    for s in ["train", "val", "test"]:
        r = cc["splits"].get(s, {}).get("REAL", 0)
        f = cc["splits"].get(s, {}).get("FAKE", 0)
        print(f"  Split '{s:<5}': {r:3d} REAL | {f:3d} FAKE | Total: {r + f:3d}")
    print("-" * 80)
    print(f"Remaining Deficit to 600/class  : REAL: {600 - cc['REAL']} | FAKE: {600 - cc['FAKE']}")
    print(f"Action Taken                    : {'QUARANTINED' if quarantine else 'AUDIT ONLY (Dry Run)'}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Strict semantic quality audit for workplace dataset")
    parser.add_argument("--quarantine", action="store_true", help="Quarantine confirmed irrelevant images to quarantine/")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    report = run_strict_semantic_audit(quarantine=args.quarantine)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_audit_summary(report, quarantine=args.quarantine)


if __name__ == "__main__":
    main()
