"""
Face Detection and Region Extraction Module
Provides fast, lightweight face localization using OpenCV.
If no face is detected, gracefully returns None to allow full-image fallback.
"""
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

# Initialize Haar Cascade Classifier from OpenCV's built-in models
_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_cascade_path)

def detect_and_crop_face(
    image: Image.Image,
    margin_ratio: float = 0.15,
    min_size: Tuple[int, int] = (50, 50),
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    """
    Detects the dominant human face in a PIL Image and returns a cropped patch.
    
    Args:
        image: Source PIL Image (RGB).
        margin_ratio: Context padding around the detected face bounding box (default 15%).
        min_size: Minimum face detection window (width, height).
        
    Returns:
        (cropped_face_pil, bbox) if a face is detected, else (None, None).
        bbox is formatted as (x1, y1, x2, y2).
    """
    if image is None:
        return None, None

    # Convert PIL Image to OpenCV grayscale format
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None, None

    # Select the largest face by area (dominant face)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face

    # Expand bounding box with contextual margin to preserve hairline & jaw boundary
    img_h, img_w = np_img.shape[:2]
    margin = int(max(w, h) * margin_ratio)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)

    # Ensure crop has valid dimensions
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None, None

    cropped_np = np_img[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped_np)

    return cropped_pil, (int(x1), int(y1), int(x2), int(y2))
