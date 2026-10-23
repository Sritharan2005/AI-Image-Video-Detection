"""
DeepGuard Video Processing & Frame Sampling Engine
===================================================
Provides robust, resource-bounded frame extraction for video deepfake analysis.
Uses OpenCV VideoCapture to sample evenly distributed frames without loading
the entire video into memory.

Strict safety constraints:
- Enforces maximum file size, duration, and sampled frame bounds.
- Cleans up VideoCapture handles and temporary files in all paths.
- Does NOT claim temporal manipulation detection; samples frames for spatial
  neural network evaluation only.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image

from app.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_FILE_SIZE_BYTES,
    MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_VIDEO_SAMPLE_FRAMES,
    MIN_VIDEO_SAMPLE_FRAMES,
    MAX_VIDEO_SAMPLE_FRAMES,
)


class VideoProcessingError(Exception):
    """Base exception for video processing failures."""
    pass


class UnsupportedVideoFormatError(VideoProcessingError):
    """Raised when the video format or extension is not supported."""
    pass


class CorruptVideoError(VideoProcessingError):
    """Raised when the video file cannot be opened or decoded."""
    pass


class VideoTooLargeError(VideoProcessingError):
    """Raised when video file size or duration exceeds safe bounds."""
    pass


def extract_video_metadata_and_frames(
    video_path: str,
    max_frames: int = DEFAULT_VIDEO_SAMPLE_FRAMES,
) -> Dict[str, Any]:
    """
    Safely opens a video file, validates duration and decodability,
    and extracts a bounded set of evenly distributed frames converted to PIL Images.
    
    Args:
        video_path: Absolute or relative path to the video file on disk.
        max_frames: Target number of frames to sample (bounded between 1 and 32).
        
    Returns:
        Dictionary containing video metadata and a list of sampled frame records:
        {
            "metadata": {
                "total_frames": int,
                "fps": float,
                "duration_seconds": float,
                "width": int,
                "height": int,
                "frames_sampled": int,
                "frames_failed": int,
            },
            "sampled_frames": [
                {
                    "frame_index": int,
                    "timestamp_seconds": float,
                    "timestamp_display": str,
                    "pil_image": Image.Image,
                    "width": int,
                    "height": int,
                }, ...
            ]
        }
    """
    if not os.path.exists(video_path):
        raise CorruptVideoError(f"Video file not found at '{video_path}'.")

    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise CorruptVideoError("Uploaded video file is empty (0 bytes).")

    if file_size > MAX_VIDEO_FILE_SIZE_BYTES:
        raise VideoTooLargeError(
            f"Video file size ({file_size // (1024*1024)} MB) exceeds "
            f"maximum permitted limit ({MAX_VIDEO_FILE_SIZE_BYTES // (1024*1024)} MB)."
        )

    # Bound max_frames
    num_samples = max(MIN_VIDEO_SAMPLE_FRAMES, min(int(max_frames), MAX_VIDEO_SAMPLE_FRAMES))

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise CorruptVideoError(
                "Unable to open video stream. The file may be corrupt or encoded with an unsupported codec."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check for invalid FPS metadata
        if fps <= 0.0 or np.isnan(fps) or np.isinf(fps):
            fps = 25.0

        # Calculate duration
        duration_seconds = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            raise VideoTooLargeError(
                f"Video duration ({duration_seconds:.1f}s) exceeds the maximum allowed limit "
                f"of {MAX_VIDEO_DURATION_SECONDS:.0f} seconds."
            )

        # Plan sample indices
        if total_frames <= 0:
            # Container missing frame count; read sequentially
            sampled_frames = []
            frames_failed = 0
            curr_idx = 0
            while len(sampled_frames) < num_samples:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                curr_time = curr_idx / fps
                if curr_time > MAX_VIDEO_DURATION_SECONDS:
                    raise VideoTooLargeError("Video duration exceeds permitted limit.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                sampled_frames.append({
                    "frame_index": curr_idx,
                    "timestamp_seconds": round(float(curr_time), 2),
                    "timestamp_display": f"{int(curr_time // 60):02d}:{curr_time % 60:05.2f}",
                    "pil_image": pil_img,
                    "width": pil_img.width,
                    "height": pil_img.height,
                })
                curr_idx += 1
            total_frames = curr_idx
            duration_seconds = curr_idx / fps
        else:
            # Evenly distribute sample indices across total_frames
            if total_frames <= num_samples:
                target_indices = list(range(total_frames))
            else:
                target_indices = [
                    int(round(i * (total_frames - 1) / (num_samples - 1)))
                    for i in range(num_samples)
                ]
                # Ensure unique, ordered indices
                target_indices = sorted(list(dict.fromkeys(target_indices)))

            sampled_frames = []
            frames_failed = 0

            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    frames_failed += 1
                    continue

                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_sec = (msec / 1000.0) if msec > 0 else (idx / fps)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                sampled_frames.append({
                    "frame_index": idx,
                    "timestamp_seconds": round(float(timestamp_sec), 2),
                    "timestamp_display": f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}",
                    "pil_image": pil_img,
                    "width": pil_img.width,
                    "height": pil_img.height,
                })

        if not sampled_frames:
            raise CorruptVideoError(
                "Failed to decode any valid frames from the video. "
                "The video stream may be truncated or corrupt."
            )

        metadata = {
            "total_frames": total_frames,
            "fps": round(float(fps), 2),
            "duration_seconds": round(float(duration_seconds), 2),
            "width": width,
            "height": height,
            "frames_sampled": len(sampled_frames),
            "frames_failed": frames_failed if 'frames_failed' in locals() else 0,
        }

        return {
            "metadata": metadata,
            "sampled_frames": sampled_frames,
        }

    finally:
        cap.release()
