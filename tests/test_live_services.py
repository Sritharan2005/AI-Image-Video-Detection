import os
import json
import urllib.request

def test_production_servers():
    print("=== TESTING ULTRA BACKEND (Port 8005) ===")
    health_8005 = json.loads(urllib.request.urlopen("http://127.0.0.1:8005/health", timeout=5).read().decode())
    print("Ultra Backend Health:", health_8005["status"], "| Model backend:", health_8005["model"]["backend"])

    print("\n=== TESTING DEEPGUARD BACKEND (Port 8000) ===")
    status_8000 = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=5).read().decode())
    print("DeepGuard Status:", status_8000)

    # Test /predict
    boundary = "----TestBoundary"
    test_img = r"C:\Users\admin\OneDrive\Desktop\AI Fake Image and Video dedection\backend\data\validation_eval\REAL\local_real_man.jpg"
    with open(test_img, "rb") as f:
        img_bytes = f.read()
    
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="local_real_man.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode('utf-8') + img_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')
    
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        pred_data = json.loads(resp.read().decode())
        print("DeepGuard /api/predict Response:", pred_data)

    # Test /api/predict/multi-crop
    req_mc = urllib.request.Request("http://127.0.0.1:8000/api/predict/multi-crop", data=body)
    req_mc.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req_mc, timeout=10) as resp:
        mc_data = json.loads(resp.read().decode())
        print("DeepGuard /api/predict/multi-crop Response:", mc_data)

if __name__ == "__main__":
    test_production_servers()
