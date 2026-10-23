"""
FastAPI Routes for DeepGuard Prediction API
Exposes POST /predict (baseline full-image with default 0.40 threshold)
and optional experimental face-crop mode with configurable thresholds (default 0.35).
"""
import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    MODEL_WEIGHTS_PATH,
    MODEL_NAME,
    CLASS_NAMES,
    CLASS_MAPPING,
    IMG_SIZE,
    RESIZE_SIZE,
    DEFAULT_PRODUCTION_THRESHOLD,
    EXPERIMENTAL_FACE_CROP_THRESHOLD,
    EXPERIMENTAL_MULTI_CROP_THRESHOLD,
    DEFAULT_AGGREGATION_METHOD,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_FILE_SIZE_BYTES,
    DEFAULT_VIDEO_SAMPLE_FRAMES,
    MIN_VIDEO_SAMPLE_FRAMES,
    MAX_VIDEO_SAMPLE_FRAMES,
    DEFAULT_VIDEO_AGGREGATION_METHOD,
)
from app.predictor import (
    predictor,
    WeightsNotLoadedError,
    InvalidImageError,
)
from app.video_processor import (
    VideoProcessingError,
    UnsupportedVideoFormatError,
    CorruptVideoError,
    VideoTooLargeError,
)

app = FastAPI(
    title="DeepGuard AI Prediction Service",
    description="FastAPI service for real vs. AI-generated media detection using EfficientNet-B0",
    version="2.3.0",
)

# Enable CORS for local frontend development (Vite / React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB

@app.get("/")
def root():
    return {
        "service": "DeepGuard AI Prediction API",
        "model_architecture": MODEL_NAME,
        "classes": CLASS_NAMES,
        "weights_loaded": predictor.is_loaded,
        "default_thresholds": {
            "baseline_full_image": DEFAULT_PRODUCTION_THRESHOLD,
            "experimental_face_crop": EXPERIMENTAL_FACE_CROP_THRESHOLD,
            "experimental_multi_crop": EXPERIMENTAL_MULTI_CROP_THRESHOLD,
        },
        "status": "ready" if predictor.is_loaded else "awaiting_trained_weights",
    }

@app.get("/api/status")
@app.get("/status")
def get_system_status():
    """Returns model loading, hardware, and configuration diagnostics."""
    return {
        "model_name": MODEL_NAME,
        "weights_path": MODEL_WEIGHTS_PATH,
        "weights_exist": os.path.exists(MODEL_WEIGHTS_PATH),
        "weights_loaded": predictor.is_loaded,
        "device": str(predictor.device) if predictor.device else "cpu",
        "classes": CLASS_NAMES,
        "class_mapping": CLASS_MAPPING,
        "default_thresholds": {
            "baseline_full_image": DEFAULT_PRODUCTION_THRESHOLD,
            "experimental_face_crop": EXPERIMENTAL_FACE_CROP_THRESHOLD,
            "experimental_multi_crop": EXPERIMENTAL_MULTI_CROP_THRESHOLD,
        },
        "benchmark_metrics": {
            "held_out_test_accuracy": 0.768,
            "fake_recall": 0.6387,
            "fake_precision": 0.8615,
            "fake_false_negative_rate": 0.3613,
        },
        "preprocessing": {
            "resize": RESIZE_SIZE,
            "center_crop": IMG_SIZE,
            "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        },
    }

async def _validate_file(file: UploadFile) -> bytes:
    filename = file.filename or "uploaded_image"
    ext = os.path.splitext(filename)[1].lower()

    if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            ),
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty (0 bytes).",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum permitted limit ({MAX_FILE_SIZE_BYTES // (1024*1024)} MB).",
        )

    return content

@app.post("/predict")
@app.post("/api/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Image file to analyze for deepfake or AI manipulation"),
    type: Optional[str] = Form("image"),
    mode: Optional[str] = Query("baseline", description="Inference mode: 'baseline' (default full-image), 'experimental_face', or 'experimental_multi_crop'"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional custom decision threshold for P(FAKE) in [0.0, 1.0]. Defaults to 0.40 in baseline mode, 0.35 in experimental_face mode, 0.50 in multi-crop."),
    aggregation: Optional[str] = Query("mean", pattern="^(mean|max)$", description="Aggregation method for multi-crop: 'mean' or 'max'"),
):
    """
    Production inference endpoint.
    Defaults to baseline full-image mode with threshold 0.40.
    Accepts mode='experimental_face' (threshold 0.35 default) or mode='experimental_multi_crop'.
    Clearly separates raw model probabilities from threshold-based classification.
    """
    content = await _validate_file(file)

    if not predictor.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "weights_not_loaded",
                "detail": f"Model weights have not been loaded from '{MODEL_WEIGHTS_PATH}'.",
                "model_architecture": MODEL_NAME,
                "trained": False,
            },
        )

    try:
        use_face_crop = (mode == "experimental_face")
        inference_result = predictor.predict_image_bytes(
            content,
            use_face_crop=use_face_crop,
            fake_threshold=threshold,
            mode=mode,
            aggregation=aggregation,
        )
        return {
            "status": "success",
            "filename": file.filename or "uploaded_image",
            **inference_result,
        }
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except WeightsNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")

@app.post("/predict/multi-crop")
@app.post("/api/predict/multi-crop")
async def predict_multi_crop_endpoint(
    file: UploadFile = File(..., description="Image file to analyze with multi-crop spatial evaluation"),
    threshold: Optional[float] = Query(EXPERIMENTAL_MULTI_CROP_THRESHOLD, ge=0.0, le=1.0),
    aggregation: Optional[str] = Query("mean", pattern="^(mean|max)$"),
):
    """
    Dedicated endpoint for experimental multi-crop evaluation.
    Evaluates global, directional (top/center/bottom or left/center/right), and face crops.
    """
    content = await _validate_file(file)

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights are not loaded.",
        )

    try:
        result = predictor.predict_multi_crop(
            content,
            fake_threshold=threshold,
            aggregation=aggregation,
        )
        return {
            "status": "success",
            "filename": file.filename or "uploaded_image",
            **result,
        }
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Multi-crop error: {str(e)}")

@app.post("/predict/diagnostic")
@app.post("/api/predict/diagnostic")
async def predict_diagnostic_endpoint(
    file: UploadFile = File(..., description="Image file for diagnostic baseline vs face-crop vs multi-crop evaluation"),
    baseline_threshold: Optional[float] = Query(DEFAULT_PRODUCTION_THRESHOLD, ge=0.0, le=1.0),
    face_crop_threshold: Optional[float] = Query(EXPERIMENTAL_FACE_CROP_THRESHOLD, ge=0.0, le=1.0),
    multi_crop_threshold: Optional[float] = Query(EXPERIMENTAL_MULTI_CROP_THRESHOLD, ge=0.0, le=1.0),
    aggregation: Optional[str] = Query("mean", pattern="^(mean|max)$"),
):
    """
    Diagnostic endpoint.
    Runs baseline full-image, experimental face-crop, and experimental multi-crop side-by-side
    with their respective decision thresholds.
    """
    content = await _validate_file(file)

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights are not loaded.",
        )

    try:
        diagnostic_result = predictor.predict_diagnostic(
            content,
            baseline_threshold=baseline_threshold,
            face_crop_threshold=face_crop_threshold,
            multi_crop_threshold=multi_crop_threshold,
            aggregation=aggregation,
        )
        return {
            "status": "success",
            "filename": file.filename or "diagnostic_image",
            **diagnostic_result,
        }
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Diagnostic error: {str(e)}")

@app.post("/predict/video")
@app.post("/api/predict/video")
async def predict_video_endpoint(
    file: UploadFile = File(..., description="Video file to analyze with frame-based evaluation"),
    threshold: Optional[float] = Query(DEFAULT_PRODUCTION_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold for P(FAKE) in [0.0, 1.0]. Defaults to 0.40."),
    max_frames: Optional[int] = Query(DEFAULT_VIDEO_SAMPLE_FRAMES, ge=MIN_VIDEO_SAMPLE_FRAMES, le=MAX_VIDEO_SAMPLE_FRAMES, description="Number of frames to sample evenly across video duration."),
    aggregation: Optional[str] = Query(DEFAULT_VIDEO_AGGREGATION_METHOD, pattern="^(mean|max)$", description="Aggregation method for per-frame scores: 'mean' or 'max'"),
):
    """
    Dedicated video inference endpoint.
    Extracts evenly distributed frames across the video and evaluates each frame with EfficientNet-B0.
    Produces an overall video summary using the specified aggregation method.

    Strict notice:
    This endpoint performs static frame-level spatial feature evaluation only.
    It does not detect temporal manipulation or cross-frame artifacts.
    """
    filename = file.filename or "uploaded_video.mp4"
    ext = os.path.splitext(filename)[1].lower()

    if not ext or ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported video format '{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            ),
        )

    if not predictor.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "weights_not_loaded",
                "detail": f"Model weights have not been loaded from '{MODEL_WEIGHTS_PATH}'.",
                "model_architecture": MODEL_NAME,
                "trained": False,
            },
        )

    # Securely stream into a temporary file with bounded file size limit
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="deepguard_video_")
    try:
        total_bytes = 0
        with os.fdopen(temp_fd, "wb") as f_out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video file size exceeds maximum permitted limit ({MAX_VIDEO_FILE_SIZE_BYTES // (1024*1024)} MB).",
                    )
                f_out.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded video file is empty (0 bytes).",
            )

        video_result = predictor.predict_video(
            video_path=temp_path,
            fake_threshold=threshold,
            max_frames=max_frames,
            aggregation=aggregation,
        )

        return {
            "status": "success",
            "filename": filename,
            **video_result,
        }
    except VideoTooLargeError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except (CorruptVideoError, InvalidImageError) as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except WeightsNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis error: {str(e)}",
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
