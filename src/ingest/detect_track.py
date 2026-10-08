import cv2
import yaml
import torch
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from src.utils.vram import check_vram_headroom


@dataclass
class TrackObservation:
    frame_idx: int
    offset_s: float
    iso_time: str
    bbox_px: List[int]        # [x1, y1, x2, y2]
    bbox_norm: List[float]    # [x1/W, y1/H, x2/W, y2/H]
    conf: float
    score: float              # area_px * conf
    frame_bgr: np.ndarray


@dataclass
class TrackSummary:
    track_id: int
    label: str
    t_start: str
    t_end: str
    t_best: str
    offset_start: float
    offset_end: float
    offset_best: float
    bbox_px: List[int]
    bbox_norm: List[float]
    best_crop: np.ndarray
    best_snapshot_bgr: np.ndarray


@dataclass
class SampledFrame:
    frame_idx: int
    offset_s: float
    iso_time: str
    frame_bgr: np.ndarray


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Normalize video orientation by rotating frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def load_yolo_world(
    weights_path: str = "yolov8s-worldv2.pt",
    vocab_path: str = "config/vocab.yaml",
    device: str = "cuda:0"
) -> Tuple[YOLO, List[str]]:
    """
    Load YOLO-World with classes configured on CPU to prevent ~443 MiB text encoder VRAM leak,
    then move vision detector to GPU.
    """
    # Guard check VRAM headroom
    if device.startswith("cuda"):
        if not check_vram_headroom(required_free_mib=1000):
            print("Warning: Low VRAM detected during YOLO load, proceeding with caution")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = yaml.safe_load(f)
    classes = vocab_data["classes"]

    # Load on CPU first
    yolo = YOLO(weights_path)
    # Set classes on CPU
    yolo.set_classes(classes)
    
    # Move model to target device
    if device.startswith("cuda") and torch.cuda.is_available():
        yolo.to(device)
    
    return yolo, classes


def reset_tracker(yolo_model: YOLO) -> None:
    """
    Explicitly reset ByteTrack tracker state to prevent track ID leakage across videos.
    """
    try:
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack._count = 0
    except Exception:
        pass
        
    try:
        from ultralytics.trackers.byte_tracker import STrack
        STrack.reset_id()
    except Exception:
        pass
        
    if hasattr(yolo_model, "predictor") and yolo_model.predictor is not None:
        if hasattr(yolo_model.predictor, "trackers") and yolo_model.predictor.trackers:
            for trk in yolo_model.predictor.trackers:
                if hasattr(trk, "reset"):
                    trk.reset()


def create_annotated_snapshot(
    frame: np.ndarray,
    bbox_px: List[int],
    label: str,
    track_id: int,
    conf: float,
    max_dim: int = 640
) -> np.ndarray:
    """
    Draw bounding box and label on snapshot, then resize proportionally to max_dim.
    """
    annotated = frame.copy()
    x1, y1, x2, y2 = bbox_px
    h, w = frame.shape[:2]
    
    # Draw box
    color = (0, 255, 0)
    thickness = max(2, int(min(w, h) / 300))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    
    # Text label
    text = f"ID{track_id} {label} {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 1000.0)
    txt_thick = max(1, thickness - 1)
    
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, txt_thick)
    bg_y1 = max(0, y1 - th - baseline - 4)
    bg_y2 = y1
    cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 6, bg_y2), color, -1)
    cv2.putText(annotated, text, (x1 + 3, y1 - baseline - 2), font, font_scale, (0, 0, 0), txt_thick, cv2.LINE_AA)
    
    # Resize for storage efficiency (~640px)
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        annotated = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    return annotated


def process_video_tracks(
    video_path: str | Path,
    yolo_model: YOLO,
    classes: List[str],
    start_time_iso: str,
    fps: float,
    duration_s: float,
    stride: int = 5,
    conf_thresh: float = 0.25,
    imgsz: int = 640,
    rotation: int = 0,
    frame_sample_interval_s: float = 2.0
) -> Tuple[List[TrackSummary], List[SampledFrame]]:
    """
    Process video with configurable stride, rotate frames if needed, track with ByteTrack,
    and aggregate track summaries and whole frames.
    """
    # 1. Reset tracker state for multi-video isolation
    reset_tracker(yolo_model)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
        
    try:
        base_dt = datetime.datetime.fromisoformat(start_time_iso)
    except Exception:
        base_dt = datetime.datetime.now()
        
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    
    tracks_data: Dict[int, List[TrackObservation]] = {}
    track_labels: Dict[int, str] = {}
    sampled_frames: List[SampledFrame] = []
    
    frame_idx = 0
    last_frame_sample_time = -999.0
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
            
        offset_s = frame_idx / actual_fps
        current_dt = base_dt + datetime.timedelta(seconds=offset_s)
        iso_time = current_dt.isoformat()
        
        # Apply rotation normalization if needed
        frame = rotate_frame(raw_frame, rotation) if rotation != 0 else raw_frame
        h, w = frame.shape[:2]
        
        # Whole frame sampling (~1 frame every 2s)
        if (offset_s - last_frame_sample_time) >= frame_sample_interval_s:
            sampled_frames.append(SampledFrame(
                frame_idx=frame_idx,
                offset_s=round(offset_s, 3),
                iso_time=iso_time,
                frame_bgr=frame.copy()
            ))
            last_frame_sample_time = offset_s
            
        # Detection + Tracking with stride
        if frame_idx % stride == 0:
            results = yolo_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_thresh,
                imgsz=imgsz,
                verbose=False
            )[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                for b in results.boxes:
                    if b.id is None:
                        continue
                    track_id = int(b.id[0].item())
                    cls_id = int(b.cls[0].item())
                    label = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                    conf = float(b.conf[0].item())
                    
                    xyxy = b.xyxy[0].cpu().numpy()
                    x1 = max(0, int(xyxy[0]))
                    y1 = max(0, int(xyxy[1]))
                    x2 = min(w, int(xyxy[2]))
                    y2 = min(h, int(xyxy[3]))
                    
                    # Ignore zero-area boxes
                    area = (x2 - x1) * (y2 - y1)
                    if area <= 0:
                        continue
                        
                    bbox_norm = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
                    score = area * conf
                    
                    obs = TrackObservation(
                        frame_idx=frame_idx,
                        offset_s=round(offset_s, 3),
                        iso_time=iso_time,
                        bbox_px=[x1, y1, x2, y2],
                        bbox_norm=bbox_norm,
                        conf=conf,
                        score=score,
                        frame_bgr=frame.copy()
                    )
                    
                    if track_id not in tracks_data:
                        tracks_data[track_id] = []
                        track_labels[track_id] = label
                    tracks_data[track_id].append(obs)
                    
        frame_idx += 1
        
    cap.release()
    
    # 2. Select best frame per track and build TrackSummary
    summaries: List[TrackSummary] = []
    for track_id, observations in tracks_data.items():
        if not observations:
            continue
            
        # Best observation has maximum score (area * conf)
        best_obs = max(observations, key=lambda o: o.score)
        first_obs = observations[0]
        last_obs = observations[-1]
        
        # Crop best object
        bx1, by1, bx2, by2 = best_obs.bbox_px
        crop = best_obs.frame_bgr[by1:by2, bx1:bx2].copy()
        
        # Annotated snapshot
        label = track_labels[track_id]
        snapshot = create_annotated_snapshot(
            best_obs.frame_bgr,
            best_obs.bbox_px,
            label,
            track_id,
            best_obs.conf,
            max_dim=640
        )
        
        summaries.append(TrackSummary(
            track_id=track_id,
            label=label,
            t_start=first_obs.iso_time,
            t_end=last_obs.iso_time,
            t_best=best_obs.iso_time,
            offset_start=first_obs.offset_s,
            offset_end=last_obs.offset_s,
            offset_best=best_obs.offset_s,
            bbox_px=best_obs.bbox_px,
            bbox_norm=best_obs.bbox_norm,
            best_crop=crop,
            best_snapshot_bgr=snapshot
        ))
        
    # Reset tracker after processing video
    reset_tracker(yolo_model)
    return summaries, sampled_frames
