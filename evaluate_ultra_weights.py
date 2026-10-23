import os
import json
import urllib.request
from PIL import Image
from aidetector.model import UltraDetector

val_dir = r'C:\Users\admin\OneDrive\Desktop\AI Fake Image and Video dedection\backend\data\validation_eval'
fake_files = [os.path.join(val_dir, 'FAKE', f) for f in os.listdir(os.path.join(val_dir, 'FAKE')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
real_files = [os.path.join(val_dir, 'REAL', f) for f in os.listdir(os.path.join(val_dir, 'REAL')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

samples = [(f, 1) for f in fake_files] + [(r, 0) for r in real_files] # 1=FAKE, 0=REAL
total_pos = len(fake_files)
total_neg = len(real_files)

print(f"Dataset: {len(samples)} samples ({total_pos} FAKE, {total_neg} REAL)")

print("Instantiating UltraDetector once...")
ultra = UltraDetector(threshold=0.50)

def post_multi_crop(img_path, agg):
    url = f"http://127.0.0.1:8000/api/predict/multi-crop?aggregation={agg}"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    fname = os.path.basename(img_path)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode('utf-8') + img_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))

eval_results = {
    'Ultra (20/80) [Production]': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    'Ultra (50/50)': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    'Ultra (80/20)': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    'MultiCrop (Mean)': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    'MultiCrop (Max)': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
}

for idx, (path, y_true) in enumerate(samples, 1):
    print(f"[{idx}/{len(samples)}] Evaluating {os.path.basename(path)} (y_true={y_true})...")
    with Image.open(path) as im:
        im_rgb = im.convert('RGB')
        hp_res = ultra.primary.predict_image(im_rgb)
        sentry_res = ultra.secondary.predict_image(im_rgb)
        
        p_hp = hp_res.probability_ai
        p_sen = sentry_res.probability_ai
        
        p_20_80 = 0.20 * p_hp + 0.80 * p_sen
        p_50_50 = 0.50 * p_hp + 0.50 * p_sen
        p_80_20 = 0.80 * p_hp + 0.20 * p_sen
        
        preds = {
            'Ultra (20/80) [Production]': 1 if p_20_80 >= 0.50 else 0,
            'Ultra (50/50)': 1 if p_50_50 >= 0.50 else 0,
            'Ultra (80/20)': 1 if p_80_20 >= 0.50 else 0,
        }
        
    try:
        mc_mean_resp = post_multi_crop(path, 'mean')
        preds['MultiCrop (Mean)'] = 1 if mc_mean_resp.get('prediction') == 'FAKE' else 0
    except Exception as e:
        preds['MultiCrop (Mean)'] = None

    try:
        mc_max_resp = post_multi_crop(path, 'max')
        preds['MultiCrop (Max)'] = 1 if mc_max_resp.get('prediction') == 'FAKE' else 0
    except Exception as e:
        preds['MultiCrop (Max)'] = None

    for m_name, y_pred in preds.items():
        if y_pred is None:
            continue
        if y_true == 1 and y_pred == 1:
            eval_results[m_name]['tp'] += 1
        elif y_true == 0 and y_pred == 1:
            eval_results[m_name]['fp'] += 1
        elif y_true == 0 and y_pred == 0:
            eval_results[m_name]['tn'] += 1
        elif y_true == 1 and y_pred == 0:
            eval_results[m_name]['fn'] += 1

print("\n=== VALIDATION DATA RESULTS (N=25: 13 FAKE, 12 REAL) ===")
summary_rows = []
for m_name, counts in eval_results.items():
    tp = counts['tp']
    fp = counts['fp']
    tn = counts['tn']
    fn = counts['fn']
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    fnr = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    summary_rows.append({
        'Model': m_name,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'FPR': round(fpr, 2),
        'FNR': round(fnr, 2),
        'F1': round(f1 / 100.0, 4)
    })
    print(f"{m_name}: TP={tp}, FP={fp}, TN={tn}, FN={fn} | Prec={precision:.1f}%, Rec={recall:.1f}%, FPR={fpr:.1f}%, FNR={fnr:.1f}%, F1={f1/100.0:.4f}")

# Test suspected image
suspected_img_path = r'C:\Users\admin\Downloads\home page of portfolio.png'
if os.path.exists(suspected_img_path):
    print(f"\n=== SUSPECTED TEST IMAGE: {os.path.basename(suspected_img_path)} ===")
    with Image.open(suspected_img_path) as im:
        im_rgb = im.convert('RGB')
        hp_res = ultra.primary.predict_image(im_rgb)
        sentry_res = ultra.secondary.predict_image(im_rgb)
        p_hp = hp_res.probability_ai
        p_sen = sentry_res.probability_ai
        print(f"Hybrid-Plus: P(AI) = {p_hp:.4f}")
        print(f"Sentry:      P(AI) = {p_sen:.4f}")
        print(f"Ultra (20/80): P(AI) = {0.20*p_hp + 0.80*p_sen:.4f} -> {'AI' if (0.20*p_hp + 0.80*p_sen)>=0.5 else 'HUMAN'}")
        print(f"Ultra (50/50): P(AI) = {0.50*p_hp + 0.50*p_sen:.4f} -> {'AI' if (0.50*p_hp + 0.50*p_sen)>=0.5 else 'HUMAN'}")
        print(f"Ultra (80/20): P(AI) = {0.80*p_hp + 0.20*p_sen:.4f} -> {'AI' if (0.80*p_hp + 0.20*p_sen)>=0.5 else 'HUMAN'}")
    
    try:
        mc_m = post_multi_crop(suspected_img_path, 'mean')
        print(f"MultiCrop (Mean): {mc_m.get('prediction')} (P(FAKE)={mc_m.get('fake_probability', 0):.4f})")
    except Exception as e:
        print(f"MultiCrop (Mean): Error {e}")
        
    try:
        mc_x = post_multi_crop(suspected_img_path, 'max')
        print(f"MultiCrop (Max):  {mc_x.get('prediction')} (P(FAKE)={mc_x.get('fake_probability', 0):.4f})")
    except Exception as e:
        print(f"MultiCrop (Max): Error {e}")

with open('eval_summary.json', 'w') as f:
    json.dump(summary_rows, f, indent=2)
print("\nSaved eval_summary.json")
