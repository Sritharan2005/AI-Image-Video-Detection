import urllib.request
import json
from collections import Counter
from PIL import Image
import io

def inspect_openfake():
    print("--- Querying ComplexDataLab/OpenFake Metadata ---")
    splits = ["train", "validation", "test"]
    headers = {"User-Agent": "DeepGuard-Audit/1.0"}

    # 1. Inspect dataset info / split sizes
    info_url = "https://datasets-server.huggingface.co/info?dataset=ComplexDataLab/OpenFake"
    try:
        req = urllib.request.Request(info_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            info = json.loads(resp.read().decode())
            dataset_info = info.get("dataset_info", {}).get("core", {})
            print("Splits and counts in 'core' config:")
            for sp, sp_info in dataset_info.get("splits", {}).items():
                print(f"  Split '{sp}': {sp_info.get('num_examples')} examples ({sp_info.get('num_bytes') / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"Failed to fetch dataset info: {e}")

    # 2. Sample 100 rows from train split to inspect generators, sources, dimensions, and prompts
    sample_url = "https://datasets-server.huggingface.co/rows?dataset=ComplexDataLab/OpenFake&config=core&split=train&offset=0&limit=100"
    try:
        req = urllib.request.Request(sample_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            rows = data.get("rows", [])
            print(f"\nSuccessfully fetched {len(rows)} sample records from 'train' split.")
            if rows:
                print("Available columns in row:", list(rows[0]["row"].keys()))
                
                label_counts = Counter()
                source_counts = Counter()
                model_counts = Counter()
                aspect_ratios = []
                dimensions = []
                prompts = []
                
                for r in rows:
                    row = r["row"]
                    lbl = str(row.get("label", "unknown")).lower()
                    label_counts[lbl] += 1
                    
                    src = row.get("source", "unknown")
                    source_counts[src] += 1
                    
                    mdl = row.get("model") or row.get("generator") or row.get("method") or "not_specified"
                    model_counts[mdl] += 1
                    
                    prm = row.get("prompt") or row.get("caption") or row.get("text") or ""
                    if prm:
                        prompts.append((lbl, str(prm)[:80]))

                print("\n--- Distribution in 100 Sample Rows ---")
                print("Labels:", dict(label_counts))
                print("Sources:", dict(source_counts.most_common(10)))
                print("Models / Generators:", dict(model_counts.most_common(10)))

                # Check dimensions and formats of first 10 sample images
                print("\n--- Sample Image Dimensions & Formats ---")
                for i, r in enumerate(rows[:10]):
                    row = r["row"]
                    img_dict = row.get("image", {})
                    img_url = img_dict.get("src")
                    if img_url:
                        try:
                            img_req = urllib.request.Request(img_url, headers=headers)
                            with urllib.request.urlopen(img_req, timeout=10) as img_resp:
                                img_bytes = img_resp.read()
                                im = Image.open(io.BytesIO(img_bytes))
                                print(f"  [{row.get('label').upper():<4}] Model: {row.get('model', 'unknown'):<18} | Size: {im.width}x{im.height} ({im.width/im.height:.2f} AR) | Format: {im.format} | Mode: {im.mode}")
                        except Exception as e:
                            print(f"  Failed to load image {i}: {e}")

                # Check if prompt / caption mentions office, portrait, man, laptop, workplace
                office_matches = [p for p in prompts if any(w in p[1].lower() for w in ["office", "laptop", "workplace", "desk", "computer", "businessman", "worker"])]
                portrait_matches = [p for p in prompts if any(w in p[1].lower() for w in ["portrait", "face", "man", "woman", "person"])]
                print(f"\nPrompts mentioning office/laptop/desk: {len(office_matches)} / {len(prompts)}")
                for lbl, p in office_matches[:5]:
                    print(f"  [{lbl.upper()}] {p}")
                print(f"Prompts mentioning portrait/face/person: {len(portrait_matches)} / {len(prompts)}")
                for lbl, p in portrait_matches[:5]:
                    print(f"  [{lbl.upper()}] {p}")
                    
    except Exception as e:
        print(f"Failed to fetch sample rows: {e}")

if __name__ == "__main__":
    inspect_openfake()

