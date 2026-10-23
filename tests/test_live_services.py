import os
import json
import urllib.request
from pathlib import Path
import pytest

@pytest.mark.service
def test_production_servers():
    print("=== TESTING ULTRA BACKEND (Port 8005) ===")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8005/health", timeout=5) as resp:
            health_8005 = json.loads(resp.read().decode())
        print("Ultra Backend Health:", health_8005.get("status"), "| Model backend:", health_8005.get("model", {}).get("backend"))
        assert health_8005.get("status") == "ok"
    except Exception as exc:
        pytest.fail(f"Ultra Backend on port 8005 unreachable: {exc}")

    print("\n=== TESTING DEEPGUARD BACKEND (Port 8000) ===")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=5) as resp:
            status_8000 = json.loads(resp.read().decode())
        print("DeepGuard Status:", status_8000.get("model_name"))
        assert "model_name" in status_8000
    except Exception as exc:
        pytest.fail(f"DeepGuard Backend on port 8000 unreachable: {exc}")

    # Test /predict
    boundary = "----TestBoundary"
    primary_img = r"C:\Users\admin\OneDrive\Desktop\AI Fake Image and Video dedection\backend\data\validation_eval\REAL\local_real_man.jpg"
    fallback_img = Path(__file__).resolve().parent.parent / "test_images" / "ai-generated.png"
    if os.path.exists(primary_img):
        img_source = Path(primary_img)
    elif fallback_img.exists():
        img_source = fallback_img
    else:
        pytest.fail("Neither primary validation image nor fallback test_images/ai-generated.png was found.")

    with open(img_source, "rb") as f:
        img_bytes = f.read()

    mime_type = "image/png" if img_source.suffix.lower() == ".png" else "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{img_source.name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode('utf-8') + img_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        pred_data = json.loads(resp.read().decode())
        print("DeepGuard /api/predict Response:", pred_data)
        assert pred_data is not None

    # Test /api/predict/multi-crop
    req_mc = urllib.request.Request("http://127.0.0.1:8000/api/predict/multi-crop", data=body)
    req_mc.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req_mc, timeout=10) as resp:
        mc_data = json.loads(resp.read().decode())
        print("DeepGuard /api/predict/multi-crop Response:", mc_data)
        assert mc_data is not None

if __name__ == "__main__":
    test_production_servers()
