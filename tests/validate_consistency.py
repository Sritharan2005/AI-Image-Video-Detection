import io
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from aidetector.model import create_detector
from aidetector.api import create_app

def run_consistency_check():
    print("=== VALIDATING PREDICTION CONSISTENCY: DIRECT MODEL vs BACKEND API ===")
    
    detector = create_detector("univfd", threshold=0.5)
    app = create_app(backend="univfd", threshold=0.5)
    client = TestClient(app)
    
    test_dir = Path("test_images")
    test_files = ["ai-generated.png", "ai_retouched.png", "human.jpeg"]
    
    for filename in test_files:
        filepath = test_dir / filename
        assert filepath.exists(), f"Missing test file {filepath}"
        
        # 1. Direct Model Prediction
        with Image.open(filepath) as img:
            direct_result = detector.predict_image(img)
            
        # 2. API Endpoint Prediction (POST /detect)
        with open(filepath, "rb") as f:
            file_bytes = f.read()
            
        api_resp = client.post(
            "/detect",
            files={"file": (filename, file_bytes, "image/png" if filename.endswith(".png") else "image/jpeg")}
        )
        assert api_resp.status_code == 200, f"API error: {api_resp.text}"
        api_data = api_resp.json()
        
        # Compare
        print(f"\n--- Test Image: {filename} ---")
        print(f"Direct: Label={direct_result.label}, P(AI)={direct_result.probability_ai:.6f}, Conf={direct_result.confidence:.6f}, Raw={direct_result.raw_score:.6f}")
        print(f"API:    Label={api_data['label']}, P(AI)={api_data['probability_ai']:.6f}, Conf={api_data['confidence']:.6f}, Raw={api_data['raw_score']:.6f}")
        
        assert direct_result.label == api_data["label"], f"Label mismatch for {filename}"
        assert round(direct_result.probability_ai, 6) == api_data["probability_ai"], f"P(AI) mismatch for {filename}"
        assert round(direct_result.probability_real, 6) == api_data["probability_real"], f"P(REAL) mismatch for {filename}"
        assert round(direct_result.confidence, 6) == api_data["confidence"], f"Confidence mismatch for {filename}"
        assert round(direct_result.raw_score, 6) == api_data["raw_score"], f"Raw score mismatch for {filename}"
        assert direct_result.backend == api_data["backend"], f"Backend mismatch for {filename}"
        print(f"Result: EXACT MATCH (Pass)")

    print("\nALL TEST CASES MATCH DIRECT MODEL PREDICTIONS WITH 100% CONSISTENCY!")

if __name__ == "__main__":
    run_consistency_check()
