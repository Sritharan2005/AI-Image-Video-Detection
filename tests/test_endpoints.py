import urllib.request
import json
from pathlib import Path

img_path = 'test_images/ai-generated.png'
with open(img_path, 'rb') as f:
    img_bytes = f.read()

boundary = '----TestBoundary12345'

def send_post(url, mode=None):
    if mode:
        full_url = f'{url}?mode={mode}'
    else:
        full_url = url
    header_part = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="ai-generated.png"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode('utf-8')
    footer_part = f'\r\n--{boundary}--\r\n'.encode('utf-8')
    body = header_part + img_bytes + footer_part
    
    req = urllib.request.Request(full_url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))

print('=== 1. BASELINE (Port 8000) ===')
res_baseline = send_post('http://127.0.0.1:8000/predict', 'baseline')
print(json.dumps(res_baseline, indent=2))

print('\n=== 2. FACE-CROP (Port 8000) ===')
res_face_crop = send_post('http://127.0.0.1:8000/predict', 'experimental_face')
print(json.dumps(res_face_crop, indent=2))

print('\n=== 3. ULTRA (Port 8005) ===')
res_ultra = send_post('http://127.0.0.1:8005/detect')
print(json.dumps(res_ultra, indent=2))
