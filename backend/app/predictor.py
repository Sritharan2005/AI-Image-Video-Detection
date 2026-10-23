"""
DeepGuard Inference Predictor Engine
Performs real PyTorch forward passes and probability calculations.
Keeps the trained EfficientNet-B0 model loaded in memory for fast inference.
Supports both baseline full-image inference and experimental face-crop evaluation
with configurable decision thresholds.
Clearly distinguishes raw model probabilities from the threshold-based decision.
"""
import io
import os
from typing import Dict, Any, Optional
from PIL import Image

from app.config import (
    MODEL_WEIGHTS_PATH,
    CLASS_NAMES,
    CLASS_MAPPING,
    DEVICE_NAME,
    DEFAULT_PRODUCTION_THRESHOLD,
    EXPERIMENTAL_FACE_CROP_THRESHOLD,
    EXPERIMENTAL_MULTI_CROP_THRESHOLD,
    DEFAULT_AGGREGATION_METHOD,
    DEFAULT_VIDEO_SAMPLE_FRAMES,
    DEFAULT_VIDEO_AGGREGATION_METHOD,
)
from app.model import build_efficientnet_b0
from app.transforms import get_inference_transforms
from app.face_detector import detect_and_crop_face
from app.multi_crop import generate_multi_crops, aggregate_crop_probabilities
from app.video_processor import extract_video_metadata_and_frames

class WeightsNotLoadedError(Exception):
    """Raised when model weights have not yet been trained or loaded."""
    pass

class InvalidImageError(ValueError):
    """Raised when image bytes cannot be decoded or are invalid."""
    pass

class DeepGuardPredictor:
    def __init__(self, weights_path: str = MODEL_WEIGHTS_PATH):
        self.weights_path = weights_path
        self.device = None
        self.model = None
        self.transform = None
        self.is_loaded = False
        self.torch = None
        self._init_engine()

    def _init_engine(self):
        """Attempts to initialize PyTorch device, transforms, and loads trained checkpoint."""
        try:
            import torch
            self.torch = torch
            self.device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")
            self.transform = get_inference_transforms()

            # Verify checkpoint existence
            if not os.path.exists(self.weights_path):
                print(
                    f"[DeepGuard] Warning: Checkpoint file not found at '{self.weights_path}'. "
                    "Inference will return 503 until valid weights are provided."
                )
                self.is_loaded = False
                return

            print(f"[DeepGuard] Loading trained model weights from: {self.weights_path}")
            checkpoint = torch.load(self.weights_path, map_location=self.device)

            # Inspect checkpoint format before loading
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
            elif isinstance(checkpoint, dict):
                if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

                # Strip potential 'module.' prefix if saved with DataParallel
                cleaned_state_dict = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }

                # Build model architecture matching training code
                self.model = build_efficientnet_b0(pretrained=False, num_classes=len(CLASS_NAMES))
                missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=True)
                if missing or unexpected:
                    print(f"[DeepGuard] Warning: Keys mismatch - missing: {missing}, unexpected: {unexpected}")
            else:
                raise TypeError(f"Unrecognized checkpoint object type: {type(checkpoint)}")

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[DeepGuard] EfficientNet-B0 loaded onto {self.device} and ready in memory.")
        except ImportError:
            print("[DeepGuard] Warning: PyTorch / torchvision is not installed in the environment.")
            self.is_loaded = False
        except Exception as e:
            print(f"[DeepGuard] Error loading weights from '{self.weights_path}': {e}")
            self.is_loaded = False

    def _run_forward(self, pil_image: Image.Image) -> Dict[str, float]:
        """Runs inference on a prepared PIL RGB image and returns raw class probabilities."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1).squeeze(0)

        # Index 0: FAKE, Index 1: REAL
        prob_map = {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(len(CLASS_NAMES))
        }

        return {
            "REAL": prob_map.get("REAL", 0.0),
            "FAKE": prob_map.get("FAKE", 0.0),
        }

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        use_face_crop: bool = False,
        fake_threshold: Optional[float] = None,
        mode: Optional[str] = None,
        aggregation: str = DEFAULT_AGGREGATION_METHOD,
    ) -> Dict[str, Any]:
        """
        Runs genuine neural network inference on raw image bytes.
        
        Args:
            image_bytes: Raw binary image payload.
            use_face_crop: If True, detects face and runs inference on face crop.
                           If False or if no face is detected, runs on full image.
            fake_threshold: Decision threshold for FAKE class in [0.0, 1.0].
                           Defaults to DEFAULT_PRODUCTION_THRESHOLD (0.40) in baseline mode,
                           or EXPERIMENTAL_FACE_CROP_THRESHOLD (0.35) in face-crop mode.
            mode: Optional explicit mode: 'baseline', 'experimental_face', or 'experimental_multi_crop'.
            aggregation: Aggregation method when in multi-crop mode ('mean' or 'max').
            
        Returns:
            Dictionary with prediction, confidence, raw probabilities, and threshold decision metadata.
            Clearly distinguishes raw model probabilities from threshold-based classification.
        """
        if mode in ("experimental_multi_crop", "multi_crop"):
            return self.predict_multi_crop(
                image_bytes=image_bytes,
                fake_threshold=fake_threshold,
                aggregation=aggregation,
            )

        if mode == "experimental_face":
            use_face_crop = True

        if not self.is_loaded or self.model is None or self.transform is None:
            raise WeightsNotLoadedError(
                "Trained model weights are not loaded. "
                f"Expected weights file at: {self.weights_path}."
            )

        if not image_bytes or len(image_bytes) == 0:
            raise InvalidImageError("Uploaded image file is empty (0 bytes).")

        # Determine threshold
        if fake_threshold is None:
            threshold = (
                EXPERIMENTAL_FACE_CROP_THRESHOLD
                if use_face_crop
                else DEFAULT_PRODUCTION_THRESHOLD
            )
        else:
            threshold = float(fake_threshold)
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"Invalid decision threshold '{threshold}'. Threshold must be between 0.0 and 1.0."
                )

        # Decode image safely via PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise InvalidImageError(f"Failed to decode image bytes: {str(e)}")

        if image.width < 10 or image.height < 10:
            raise InvalidImageError(
                f"Image dimensions ({image.width}x{image.height}) are too small. "
                "Minimum required resolution is 10x10 pixels."
            )

        face_crop = None
        face_bbox = None
        inference_target = "full_image"

        if use_face_crop:
            face_crop, face_bbox = detect_and_crop_face(image)
            if face_crop is not None:
                inference_target = "face_crop"

        target_image = face_crop if face_crop is not None else image
        raw_probs = self._run_forward(target_image)

        real_prob = raw_probs["REAL"]
        fake_prob = raw_probs["FAKE"]

        # Decision rule: FAKE if P(FAKE) >= threshold, else REAL
        is_fake = fake_prob >= threshold
        predicted_class = "FAKE" if is_fake else "REAL"
        confidence = fake_prob if is_fake else real_prob

        operating_mode = (
            "experimental_face_crop"
            if (use_face_crop and face_crop is not None)
            else ("experimental_face_crop_fallback" if use_face_crop else "baseline_full_image")
        )

        explanation = (
            f"EfficientNet-B0 evaluated spatial features on {inference_target.replace('_', ' ')}. "
            f"P(FAKE)={fake_prob*100:.2f}% (Threshold: {threshold*100:.1f}%) -> Classified as {predicted_class}."
        )

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "REAL": round(real_prob, 4),
                "FAKE": round(fake_prob, 4),
            },
            "raw_probabilities": {
                "REAL": round(real_prob, 4),
                "FAKE": round(fake_prob, 4),
            },
            "decision": {
                "prediction": predicted_class,
                "confidence": round(confidence, 4),
                "threshold_applied": round(threshold, 4),
                "decision_rule": f"P(FAKE) >= {threshold:.2f} -> FAKE, else REAL",
                "operating_mode": operating_mode,
            },
            "inference_target": inference_target,
            "face_detected": face_crop is not None,
            "face_bbox": list(face_bbox) if face_bbox else None,
            "model_version": "EfficientNet-B0 (Scale-5K Fine-Tuned)",
            "inference_device": str(self.device),
            "explanation": explanation,
        }

    def predict_multi_crop(
        self,
        image_bytes: bytes,
        fake_threshold: Optional[float] = None,
        aggregation: str = DEFAULT_AGGREGATION_METHOD,
        include_face: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs experimental multi-crop inference.
        Evaluates aspect-ratio-preserving global input plus directional crops
        (top/center/bottom for portrait, left/center/right for landscape) and localized face crop.
        Aggregates crop probabilities using configurable method ('mean' or 'max').
        """
        if not self.is_loaded or self.model is None or self.transform is None:
            raise WeightsNotLoadedError(
                "Trained model weights are not loaded. "
                f"Expected weights file at: {self.weights_path}."
            )

        if not image_bytes or len(image_bytes) == 0:
            raise InvalidImageError("Uploaded image file is empty (0 bytes).")

        threshold = (
            EXPERIMENTAL_MULTI_CROP_THRESHOLD
            if fake_threshold is None
            else float(fake_threshold)
        )
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Invalid decision threshold '{threshold}'. Threshold must be between 0.0 and 1.0."
            )

        norm_agg = aggregation.strip().lower()
        if norm_agg not in ("mean", "max"):
            raise ValueError(
                f"Invalid aggregation method '{aggregation}'. Supported: 'mean', 'max'."
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise InvalidImageError(f"Failed to decode image bytes: {str(e)}")

        if image.width < 10 or image.height < 10:
            raise InvalidImageError(
                f"Image dimensions ({image.width}x{image.height}) are too small. "
                "Minimum required resolution is 10x10 pixels."
            )

        crops = generate_multi_crops(image, include_face=include_face)

        evaluated_crops = []
        face_detected = False
        face_bbox = None

        for c in crops:
            crop_probs = self._run_forward(c["crop_image"])
            if c["crop_name"] == "face_crop":
                face_detected = True
                face_bbox = c["box"]

            evaluated_crops.append({
                "crop_name": c["crop_name"],
                "type": c["type"],
                "box": c["box"],
                "dimensions": [c["crop_image"].width, c["crop_image"].height],
                "probabilities": {
                    "REAL": round(crop_probs["REAL"], 4),
                    "FAKE": round(crop_probs["FAKE"], 4),
                },
            })

        agg_probs = aggregate_crop_probabilities(evaluated_crops, method=norm_agg)
        fake_prob = agg_probs["FAKE"]
        real_prob = agg_probs["REAL"]

        is_fake = fake_prob >= threshold
        predicted_class = "FAKE" if is_fake else "REAL"
        confidence = fake_prob if is_fake else real_prob

        explanation = (
            f"Multi-crop evaluated {len(evaluated_crops)} spatial patches with '{norm_agg}' aggregation. "
            f"P(FAKE)={fake_prob*100:.2f}% (Threshold: {threshold*100:.1f}%) -> Classified as {predicted_class}."
        )

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "REAL": round(real_prob, 4),
                "FAKE": round(fake_prob, 4),
            },
            "raw_probabilities": {
                "REAL": round(real_prob, 4),
                "FAKE": round(fake_prob, 4),
            },
            "decision": {
                "prediction": predicted_class,
                "confidence": round(confidence, 4),
                "threshold_applied": round(threshold, 4),
                "decision_rule": f"P(FAKE) >= {threshold:.2f} -> FAKE, else REAL",
                "operating_mode": "experimental_multi_crop",
                "aggregation_method": norm_agg,
            },
            "inference_target": "multi_crop",
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "crop_count": len(evaluated_crops),
            "crops_evaluated": evaluated_crops,
            "model_version": "EfficientNet-B0 (Scale-5K Fine-Tuned)",
            "inference_device": str(self.device),
            "explanation": explanation,
        }

    def predict_diagnostic(
        self,
        image_bytes: bytes,
        baseline_threshold: float = DEFAULT_PRODUCTION_THRESHOLD,
        face_crop_threshold: float = EXPERIMENTAL_FACE_CROP_THRESHOLD,
        multi_crop_threshold: float = EXPERIMENTAL_MULTI_CROP_THRESHOLD,
        aggregation: str = DEFAULT_AGGREGATION_METHOD,
    ) -> Dict[str, Any]:
        """
        Runs baseline full-image, experimental face-crop, and experimental multi-crop predictions
        side-by-side with their respective configured decision thresholds.
        """
        baseline_res = self.predict_image_bytes(
            image_bytes, use_face_crop=False, fake_threshold=baseline_threshold
        )
        face_crop_res = self.predict_image_bytes(
            image_bytes, use_face_crop=True, fake_threshold=face_crop_threshold
        )
        multi_crop_res = self.predict_multi_crop(
            image_bytes, fake_threshold=multi_crop_threshold, aggregation=aggregation
        )

        return {
            "baseline_full_image": baseline_res,
            "experimental_face_crop": face_crop_res,
            "experimental_multi_crop": multi_crop_res,
            "comparison": {
                "face_detected": face_crop_res["face_detected"],
                "prediction_changed": (
                    baseline_res["prediction"] != face_crop_res["prediction"]
                    or baseline_res["prediction"] != multi_crop_res["prediction"]
                ),
                "baseline_decision": baseline_res["decision"],
                "face_crop_decision": face_crop_res["decision"],
                "multi_crop_decision": multi_crop_res["decision"],
                "fake_probability_delta_face": round(
                    face_crop_res["raw_probabilities"]["FAKE"]
                    - baseline_res["raw_probabilities"]["FAKE"],
                    4,
                ),
                "fake_probability_delta_multi": round(
                    multi_crop_res["raw_probabilities"]["FAKE"]
                    - baseline_res["raw_probabilities"]["FAKE"],
                    4,
                ),
            },
        }

    def predict_video(
        self,
        video_path: str,
        fake_threshold: Optional[float] = None,
        max_frames: int = DEFAULT_VIDEO_SAMPLE_FRAMES,
        aggregation: str = DEFAULT_VIDEO_AGGREGATION_METHOD,
    ) -> Dict[str, Any]:
        """
        Runs frame-based inference on an uploaded video file.
        Samples evenly distributed frames without loading the entire video into memory,
        applies EfficientNet-B0 inference to each frame, and aggregates per-frame scores.

        Strict constraints:
        - Evaluates spatial frame features only (does NOT detect temporal manipulation).
        - Softmax scores represent model outputs, not calibrated probabilities.
        """
        if not self.is_loaded or self.model is None or self.transform is None:
            raise WeightsNotLoadedError(
                "Trained model weights are not loaded. "
                f"Expected weights file at: {self.weights_path}."
            )

        threshold = (
            DEFAULT_PRODUCTION_THRESHOLD
            if fake_threshold is None
            else float(fake_threshold)
        )
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Invalid decision threshold '{threshold}'. Threshold must be between 0.0 and 1.0."
            )

        norm_agg = aggregation.strip().lower()
        if norm_agg not in ("mean", "max"):
            raise ValueError(
                f"Invalid aggregation method '{aggregation}'. Supported: 'mean', 'max'."
            )

        # Extract frames and metadata safely via OpenCV
        video_data = extract_video_metadata_and_frames(video_path, max_frames=max_frames)
        metadata = video_data["metadata"]
        sampled_frames = video_data["sampled_frames"]

        per_frame_predictions = []
        for frame in sampled_frames:
            crop_probs = self._run_forward(frame["pil_image"])
            real_score = crop_probs["REAL"]
            fake_score = crop_probs["FAKE"]

            frame_is_fake = fake_score >= threshold
            frame_pred = "FAKE" if frame_is_fake else "REAL"
            frame_conf = fake_score if frame_is_fake else real_score

            per_frame_predictions.append({
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "timestamp_display": frame["timestamp_display"],
                "prediction": frame_pred,
                "confidence": round(frame_conf, 4),
                "probabilities": {
                    "REAL": round(real_score, 4),
                    "FAKE": round(fake_score, 4),
                },
                "raw_probabilities": {
                    "REAL": round(real_score, 4),
                    "FAKE": round(fake_score, 4),
                },
            })

        # Score aggregation
        if norm_agg == "mean":
            agg_fake = sum(f["raw_probabilities"]["FAKE"] for f in per_frame_predictions) / len(per_frame_predictions)
            agg_real = sum(f["raw_probabilities"]["REAL"] for f in per_frame_predictions) / len(per_frame_predictions)
        else:  # max
            agg_fake = max(f["raw_probabilities"]["FAKE"] for f in per_frame_predictions)
            agg_real = 1.0 - agg_fake

        is_fake = agg_fake >= threshold
        overall_pred = "FAKE" if is_fake else "REAL"
        overall_conf = agg_fake if is_fake else agg_real

        explanation = (
            f"Frame-based video evaluation analyzed {len(per_frame_predictions)} sampled frames with '{norm_agg}' aggregation. "
            f"Aggregate model score: P(FAKE)={agg_fake*100:.2f}% (Threshold: {threshold*100:.1f}%) -> Classified as {overall_pred}. "
            "Note: Analyzes static spatial features across sampled frames only; does not assess temporal manipulation."
        )

        return {
            "media_type": "video",
            "prediction": overall_pred,
            "confidence": round(overall_conf, 4),
            "probabilities": {
                "REAL": round(agg_real, 4),
                "FAKE": round(agg_fake, 4),
            },
            "raw_probabilities": {
                "REAL": round(agg_real, 4),
                "FAKE": round(agg_fake, 4),
            },
            "decision": {
                "prediction": overall_pred,
                "confidence": round(overall_conf, 4),
                "threshold_applied": round(threshold, 4),
                "decision_rule": f"Aggregate P(FAKE) >= {threshold:.2f} -> FAKE, else REAL",
                "operating_mode": "frame_based_video_analysis",
                "aggregation_method": norm_agg,
            },
            "video_metadata": metadata,
            "per_frame_predictions": per_frame_predictions,
            "model_version": "EfficientNet-B0 (Scale-5K Fine-Tuned)",
            "inference_device": str(self.device),
            "explanation": explanation,
        }

# Global predictor instance - loaded once in memory
predictor = DeepGuardPredictor()
