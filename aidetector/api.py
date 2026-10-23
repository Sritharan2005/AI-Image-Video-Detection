from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image

from .config import (
    DEFAULT_BACKEND,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_HYBRID_UNIVFD_WEIGHT,
    DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
    DEFAULT_ULTRA_PRIMARY_WEIGHT,
)
from .model import create_detector

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    FastAPI = None
    File = None
    UploadFile = None
    CORSMiddleware = None


def create_app(
    *,
    threshold: float = 0.5,
    device: str | None = None,
    backend: str = DEFAULT_BACKEND,
    weight_path: str | Path | None = None,
    hf_model: str = DEFAULT_HF_MODEL_ID,
    hybrid_univfd_weight: float = DEFAULT_HYBRID_UNIVFD_WEIGHT,
    hybrid_plus_primary_weight: float = DEFAULT_HYBRID_PLUS_PRIMARY_WEIGHT,
    ultra_primary_weight: float = DEFAULT_ULTRA_PRIMARY_WEIGHT,
):
    if FastAPI is None or UploadFile is None:
        raise RuntimeError("Missing dependency: install ai-image-detector[api].")

    detector = create_detector(
        backend,
        device=device,
        threshold=threshold,
        weight_path=weight_path,
        hf_model=hf_model,
        hybrid_univfd_weight=hybrid_univfd_weight,
        hybrid_plus_primary_weight=hybrid_plus_primary_weight,
        ultra_primary_weight=ultra_primary_weight,
    )
    app = FastAPI(title="AI Image Detector", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": detector.model_info()}

    @app.post("/detect")
    async def detect(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        with Image.open(io.BytesIO(content)) as image:
            result = detector.predict_image(image)
        payload = result.as_dict()
        payload["filename"] = file.filename
        return payload

    residual_detector = None

    @app.post("/detect/residual")
    async def detect_residual(file: UploadFile = File(...)) -> dict:
        nonlocal residual_detector
        if residual_detector is None:
            try:
                residual_detector = create_detector("nonescape-mini", device=device, threshold=0.5)
            except Exception as exc:
                return {
                    "filename": file.filename,
                    "error": f"Residual model unavailable: {exc}",
                    "is_experimental": True,
                    "is_production_ensemble_member": False,
                    "status": "Unavailable",
                }
        try:
            content = await file.read()
            with Image.open(io.BytesIO(content)) as image:
                result = residual_detector.predict_image(image)
            prob = result.probability_ai
            if prob is None or math.isnan(prob) or not (0.0 <= prob <= 1.0):
                label = "uncertain"
                prob = None
            else:
                label = result.label
            return {
                "filename": file.filename,
                "probability_ai": prob,
                "label": label,
                "threshold": residual_detector.threshold,
                "model": "nonescape-mini-residual",
                "is_experimental": True,
                "is_production_ensemble_member": False,
                "status": "Experimental / Unvalidated Residual Analysis",
                "warning": "This experimental residual analysis is uncalibrated and isolated from production voting.",
            }
        except Exception as exc:
            return {
                "filename": file.filename,
                "error": f"Inference error: {exc}",
                "is_experimental": True,
                "is_production_ensemble_member": False,
                "status": "Error",
            }

    return app
