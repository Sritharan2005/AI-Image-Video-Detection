"""
Configuration Module for DeepGuard Backend
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_WEIGHTS_PATH = os.environ.get(
    "DEEPGUARD_MODEL_PATH",
    str(MODELS_DIR / "efficientnet_b0_deepguard.pth")
)

# Model Settings
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 2
# Training was executed with PyTorch ImageFolder on folders ['FAKE', 'REAL']
# Alphabetical directory order produces: index 0: FAKE, index 1: REAL
CLASS_NAMES = ["FAKE", "REAL"]
CLASS_MAPPING = {0: "FAKE", 1: "REAL"}

# Preprocessing & Inference Settings matching training eval_transform:
# Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean, std)
RESIZE_SIZE = 256
IMG_SIZE = 224
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# Device Selection (CUDA GPU if available, else CPU)
DEVICE_NAME = os.environ.get("DEEPGUARD_DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

# Decision Threshold Settings
# Production baseline default: standard binary decision at P(FAKE) >= 0.40
DEFAULT_PRODUCTION_THRESHOLD = 0.40
# Experimental face-crop mode default: adjusted threshold at P(FAKE) >= 0.35
EXPERIMENTAL_FACE_CROP_THRESHOLD = 0.35
# Experimental multi-crop mode default: threshold at P(FAKE) >= 0.50
EXPERIMENTAL_MULTI_CROP_THRESHOLD = 0.50
DEFAULT_AGGREGATION_METHOD = "mean"

# Video Processing & Sampling Settings
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_VIDEO_DURATION_SECONDS = 300.0  # 5 minutes maximum
DEFAULT_VIDEO_SAMPLE_FRAMES = 8
MIN_VIDEO_SAMPLE_FRAMES = 1
MAX_VIDEO_SAMPLE_FRAMES = 32
DEFAULT_VIDEO_AGGREGATION_METHOD = "mean"

