"""
DeepGuard AI - Multi-Crop Generation and Aggregation Engine
============================================================
Implements aspect-ratio-preserving directional crops (global, top/center/bottom
for portraits; left/center/right for landscapes) plus optional localized face crops.

Provides documented, configurable score aggregation:
  - 'mean': Arithmetic average of crop probabilities (stable, lower variance).
  - 'max': Maximum fake probability across all crops (sensitive to localized artifacts).

Guarantees:
  - Standard model normalization and tensor dimensions via standard inference transforms.
  - Zero duplicate or invalid crops.
  - Graceful fallback when no face is detected.
"""

from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from app.face_detector import detect_and_crop_face

def generate_multi_crops(
    image: Image.Image,
    include_face: bool = True
) -> List[Dict[str, Any]]:
    """
    Generates an ensemble of aspect-ratio-preserving crops from an input PIL RGB image.

    Args:
        image: Source PIL Image in RGB format.
        include_face: Whether to attempt face detection and append the face crop.

    Returns:
        List of dictionaries with keys:
          - 'crop_name': Human-readable identifier ('global_center', 'top', 'center', 'bottom', 'face_crop')
          - 'crop_image': PIL Image of the crop
          - 'box': [x1, y1, x2, y2] pixel coordinates relative to the original image
          - 'type': 'global' | 'directional' | 'face'
    """
    w, h = image.size
    if w < 10 or h < 10:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for multi-crop analysis.")

    crops: List[Dict[str, Any]] = []

    # 1. Global View (Baseline aspect-preserved frame)
    crops.append({
        "crop_name": "global_center",
        "crop_image": image,
        "box": [0, 0, w, h],
        "type": "global",
    })

    aspect_ratio = h / w

    # 2. Directional Square Crops
    # If vertical / portrait (H > W by >= 15%)
    if aspect_ratio >= 1.15:
        crop_size = w
        # Top crop (captures upper body, head, face region)
        top_box = (0, 0, w, crop_size)
        crops.append({
            "crop_name": "top",
            "crop_image": image.crop(top_box),
            "box": list(top_box),
            "type": "directional",
        })

        # Center crop (captures mid torso, hands, center desk/lap)
        center_y = (h - crop_size) // 2
        center_box = (0, center_y, w, center_y + crop_size)
        crops.append({
            "crop_name": "center",
            "crop_image": image.crop(center_box),
            "box": list(center_box),
            "type": "directional",
        })

        # Bottom crop (captures lower frame, laptop keyboard, foreground table)
        bottom_y = h - crop_size
        bottom_box = (0, bottom_y, w, h)
        crops.append({
            "crop_name": "bottom",
            "crop_image": image.crop(bottom_box),
            "box": list(bottom_box),
            "type": "directional",
        })

    # If horizontal / landscape (W > H by >= 15%)
    elif (1.0 / aspect_ratio) >= 1.15:
        crop_size = h
        # Left crop
        left_box = (0, 0, crop_size, h)
        crops.append({
            "crop_name": "left",
            "crop_image": image.crop(left_box),
            "box": list(left_box),
            "type": "directional",
        })

        # Center crop
        center_x = (w - crop_size) // 2
        center_box = (center_x, 0, center_x + crop_size, h)
        crops.append({
            "crop_name": "center",
            "crop_image": image.crop(center_box),
            "box": list(center_box),
            "type": "directional",
        })

        # Right crop
        right_x = w - crop_size
        right_box = (right_x, 0, w, h)
        crops.append({
            "crop_name": "right",
            "crop_image": image.crop(right_box),
            "box": list(right_box),
            "type": "directional",
        })

    # If near square (0.95 <= aspect_ratio <= 1.05), directional crops would be identical
    # duplicates of global_center, so we avoid adding duplicate crops.

    # 3. Optional Localized Face Crop
    if include_face:
        face_crop, face_bbox = detect_and_crop_face(image)
        if face_crop is not None and face_bbox is not None:
            # Validate face crop size
            if face_crop.width >= 10 and face_crop.height >= 10:
                crops.append({
                    "crop_name": "face_crop",
                    "crop_image": face_crop,
                    "box": list(face_bbox),
                    "type": "face",
                })

    return crops

def aggregate_crop_probabilities(
    crop_evaluations: List[Dict[str, Any]],
    method: str = "mean"
) -> Dict[str, float]:
    """
    Aggregates per-crop class probabilities into an overall score.

    Args:
        crop_evaluations: List of evaluated crop dicts, each having 'probabilities': {'FAKE': float, 'REAL': float}
        method: 'mean' (arithmetic mean across all crops) or 'max' (maximum fake probability across crops)

    Returns:
        Aggregated probability dict {'REAL': float, 'FAKE': float} summing to 1.0.
    """
    if not crop_evaluations:
        raise ValueError("Cannot aggregate empty crop evaluation list.")

    norm_method = method.strip().lower()
    if norm_method not in ("mean", "max"):
        raise ValueError(f"Unsupported aggregation method '{method}'. Supported methods: 'mean', 'max'.")

    fake_probs = [c["probabilities"]["FAKE"] for c in crop_evaluations]

    if norm_method == "mean":
        agg_fake = sum(fake_probs) / len(fake_probs)
    elif norm_method == "max":
        agg_fake = max(fake_probs)
    else:
        agg_fake = sum(fake_probs) / len(fake_probs)

    # Ensure probability bounds [0.0, 1.0]
    agg_fake = max(0.0, min(1.0, agg_fake))
    agg_real = max(0.0, min(1.0, 1.0 - agg_fake))

    return {
        "REAL": round(agg_real, 4),
        "FAKE": round(agg_fake, 4),
    }
