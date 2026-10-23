import io
import math
from PIL import Image
from fastapi.testclient import TestClient
from aidetector.api import create_app

def test_api_production_and_residual_endpoints():
    app = create_app(backend="ultra")
    client = TestClient(app)

    # 1. Test health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"
    assert health_resp.json()["model"]["backend"] == "ultra"

    # Create dummy image in memory
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="blue").save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # 2. Test production /detect route
    prod_resp = client.post(
        "/detect",
        files={"file": ("test_blue.jpg", img_bytes, "image/jpeg")}
    )
    assert prod_resp.status_code == 200
    prod_data = prod_resp.json()
    assert "probability_ai" in prod_data
    assert "label" in prod_data
    assert "filename" in prod_data
    # Verify production endpoint does NOT have experimental flag
    assert "is_experimental" not in prod_data

    # 3. Test experimental /detect/residual route
    resid_resp = client.post(
        "/detect/residual",
        files={"file": ("test_blue.jpg", img_bytes, "image/jpeg")}
    )
    assert resid_resp.status_code == 200
    resid_data = resid_resp.json()
    assert resid_data["is_experimental"] is True
    assert resid_data["is_production_ensemble_member"] is False
    assert resid_data["status"] == "Experimental / Unvalidated Residual Analysis"
    assert resid_data["model"] == "nonescape-mini-residual"
    assert "warning" in resid_data
    assert 0.0 <= resid_data["probability_ai"] <= 1.0

def test_residual_invalid_image_handling():
    app = create_app(backend="ultra")
    client = TestClient(app)

    # Send corrupt non-image bytes
    corrupt_resp = client.post(
        "/detect/residual",
        files={"file": ("corrupt.jpg", b"not-an-image", "image/jpeg")}
    )
    assert corrupt_resp.status_code == 200
    data = corrupt_resp.json()
    assert data["is_experimental"] is True
    assert data["is_production_ensemble_member"] is False
    assert data["status"] == "Error"
    assert "Inference error" in data["error"]
