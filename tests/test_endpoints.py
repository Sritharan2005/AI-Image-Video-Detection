import json
import urllib.error
import urllib.request
from pathlib import Path
import pytest

BOUNDARY = '----TestBoundary12345'
SAMPLE_IMAGE = Path(__file__).resolve().parent.parent / 'test_images' / 'ai-generated.png'


def _check_service_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def send_post(url: str, mode: str | None = None):
    if not SAMPLE_IMAGE.exists():
        pytest.fail(f"Sample test image not found: {SAMPLE_IMAGE}")
    with open(SAMPLE_IMAGE, 'rb') as f:
        img_bytes = f.read()

    full_url = f'{url}?mode={mode}' if mode else url
    header_part = (
        f'--{BOUNDARY}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="ai-generated.png"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode('utf-8')
    footer_part = f'\r\n--{BOUNDARY}--\r\n'.encode('utf-8')
    body = header_part + img_bytes + footer_part

    req = urllib.request.Request(full_url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={BOUNDARY}')
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))


@pytest.mark.service
def test_endpoint_baseline():
    """Verify live baseline prediction endpoint on port 8000."""
    if not _check_service_reachable('http://127.0.0.1:8000/api/status'):
        pytest.fail("Service at http://127.0.0.1:8000 is not reachable. Live service must be started before running service tests.")
    res = send_post('http://127.0.0.1:8000/predict', 'baseline')
    assert res is not None
    assert "status" in res or "prediction" in res or "probability_ai" in res


@pytest.mark.service
def test_endpoint_face_crop():
    """Verify live experimental face-crop endpoint on port 8000."""
    if not _check_service_reachable('http://127.0.0.1:8000/api/status'):
        pytest.fail("Service at http://127.0.0.1:8000 is not reachable. Live service must be started before running service tests.")
    res = send_post('http://127.0.0.1:8000/predict', 'experimental_face')
    assert res is not None


@pytest.mark.service
def test_endpoint_ultra():
    """Verify live ultra detector endpoint on port 8005."""
    if not _check_service_reachable('http://127.0.0.1:8005/health'):
        pytest.fail("Service at http://127.0.0.1:8005 is not reachable. Live service must be started before running service tests.")
    res = send_post('http://127.0.0.1:8005/detect')
    assert res is not None


if __name__ == "__main__":
    print('=== 1. BASELINE (Port 8000) ===')
    res_baseline = send_post('http://127.0.0.1:8000/predict', 'baseline')
    print(json.dumps(res_baseline, indent=2))

    print('\n=== 2. FACE-CROP (Port 8000) ===')
    res_face_crop = send_post('http://127.0.0.1:8000/predict', 'experimental_face')
    print(json.dumps(res_face_crop, indent=2))

    print('\n=== 3. ULTRA (Port 8005) ===')
    res_ultra = send_post('http://127.0.0.1:8005/detect')
    print(json.dumps(res_ultra, indent=2))
