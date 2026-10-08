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
    group: str = "object"
    colors: str = "{}"
    quality: float = 0.0
    hits: int = 1


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
    yolo.float()
    
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


def process_video_tracks_phase1b(
    video_path: str | Path,
    detector: Any,
    start_time_iso: str,
    fps: float,
    duration_s: float,
    stride: int = 2,
    tracker_name: str = "bytetrack_tuned",
    imgsz: int = 640,
    rotation: int = 0,
    frame_sample_interval_s: float = 2.0,
    min_duration_s: float = 1.0,
    min_hits: int = 4,
    min_mean_conf: float = 0.35,
    embedder: Any = None
) -> Tuple[List[TrackSummary], List[SampledFrame], Dict[str, int], List[Dict[str, Any]]]:
    """
    Phase 1b Ingest Pipeline:
    - Hybrid Detector (COCO YOLO11 for KEEP classes + YOLO-World for non-COCO)
    - Tuned ByteTrack (or BoT-SORT) with lowered stride (2-3)
    - TrackState with class voting and quality-based best frame selection (conf * size * sharpness)
    - Pad-to-square crops preserving aspect ratio
    - Strict track filtering (min_duration 1.0s, min_hits 4, min_mean_conf 0.35)
    - Track stitching (gap 0-2s, center dist < 0.25, cosine >= 0.80)
    - CIE-Lab K-Means colors JSON column
    """
    import json
    from src.ingest.detector import get_object_group
    from src.ingest.track_engine import (
        TrackState,
        TrackObservation as EngineTrackObs,
        compute_crop_quality,
        pad_to_square,
        create_tracker,
        run_tracker_on_detections,
        filter_tracks,
        stitch_tracks
    )
    from src.utils.lab_color import extract_track_colors

    tracker = create_tracker(tracker_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    try:
        base_dt = datetime.datetime.fromisoformat(start_time_iso)
    except Exception:
        base_dt = datetime.datetime.now()

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0

    raw_tracks: Dict[int, TrackState] = {}
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

        frame = rotate_frame(raw_frame, rotation) if rotation != 0 else raw_frame
        h, w = frame.shape[:2]

        # Whole frame sampling
        if (offset_s - last_frame_sample_time) >= frame_sample_interval_s:
            fh, fw = frame.shape[:2]
            scale = min(640 / max(fw, fh), 1.0)
            if scale < 1.0:
                small_f = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
            else:
                small_f = frame.copy()
            sampled_frames.append(SampledFrame(
                frame_idx=frame_idx,
                offset_s=round(offset_s, 3),
                iso_time=iso_time,
                frame_bgr=small_f
            ))
            last_frame_sample_time = offset_s

        # Detection + Tracking with stride
        if frame_idx % stride == 0:
            dets = detector.detect(frame, imgsz=imgsz)
            tracked_dets = run_tracker_on_detections(tracker, dets, frame, w, h)

            for td in tracked_dets:
                tid = td["track_id"]
                bx1, by1, bx2, by2 = td["bbox"]
                conf = td["conf"]
                label = td["label"]

                if (bx2 - bx1) <= 0 or (by2 - by1) <= 0:
                    continue

                crop = frame[by1:by2, bx1:bx2].copy()
                quality = compute_crop_quality(crop, conf, [bx1, by1, bx2, by2], w, h)
                bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]

                obs = EngineTrackObs(
                    frame_idx=frame_idx,
                    offset_s=round(offset_s, 3),
                    iso_time=iso_time,
                    bbox_px=[bx1, by1, bx2, by2],
                    bbox_norm=bbox_norm,
                    conf=conf,
                    quality=quality,
                    crop_bgr=crop
                )

                if tid not in raw_tracks:
                    raw_tracks[tid] = TrackState(track_id=tid)
                raw_tracks[tid].add(obs, label, conf, frame_bgr=frame)

        frame_idx += 1

    cap.release()

    # 1. Filter tracks
    filtered_states, dropped_counts = filter_tracks(
        raw_tracks,
        min_duration_s=min_duration_s,
        min_hits=min_hits,
        min_mean_conf=min_mean_conf
    )

    # 2. Extract best crops, padded crops, and embeddings
    summaries: List[TrackSummary] = []

    for st in filtered_states:
        best_obs, best_q = st.get_best_observation()
        first_obs = st.observations[0]
        last_obs = st.observations[-1]
        voted_label = st.get_voted_label()
        group = get_object_group(voted_label)

        raw_crop = st.best_crop_bgr if st.best_crop_bgr is not None else best_obs.crop_bgr
        padded_crop = pad_to_square(raw_crop)

        colors_dict = extract_track_colors(padded_crop, voted_label, group)
        colors_json = json.dumps(colors_dict)

        if st.best_snapshot_bgr is not None:
            snapshot = create_annotated_snapshot(
                st.best_snapshot_bgr,
                best_obs.bbox_px,
                voted_label,
                st.track_id,
                best_obs.conf,
                max_dim=640
            )
        else:
            snapshot = padded_crop

        summaries.append(TrackSummary(
            track_id=st.track_id,
            label=voted_label,
            t_start=first_obs.iso_time,
            t_end=last_obs.iso_time,
            t_best=best_obs.iso_time,
            offset_start=first_obs.offset_s,
            offset_end=last_obs.offset_s,
            offset_best=best_obs.offset_s,
            bbox_px=best_obs.bbox_px,
            bbox_norm=best_obs.bbox_norm,
            best_crop=padded_crop,
            best_snapshot_bgr=snapshot,
            group=group,
            colors=colors_json,
            quality=round(best_q, 4),
            hits=st.hits
        ))

    # 3. Track Stitching if embedder is provided
    merge_logs: List[Dict[str, Any]] = []
    if embedder is not None and len(summaries) > 1:
        crops = [s.best_crop for s in summaries]
        embs = embedder.embed_images(crops, initial_batch_size=16)

        stitched_states, stitched_embs, merge_logs = stitch_tracks(
            filtered_states,
            embs,
            max_gap_s=2.0,
            max_center_dist=0.25,
            min_cosine_sim=0.80
        )

        # Rebuild summaries from stitched states
        if len(merge_logs) > 0:
            rebuilt_summaries: List[TrackSummary] = []
            for st in stitched_states:
                best_obs, best_q = st.get_best_observation()
                first_obs = st.observations[0]
                last_obs = st.observations[-1]
                voted_label = st.get_voted_label()
                group = get_object_group(voted_label)

                raw_crop = st.best_crop_bgr if st.best_crop_bgr is not None else best_obs.crop_bgr
                padded_crop = pad_to_square(raw_crop)

                colors_dict = extract_track_colors(padded_crop, voted_label, group)
                colors_json = json.dumps(colors_dict)

                if st.best_snapshot_bgr is not None:
                    snapshot = create_annotated_snapshot(
                        st.best_snapshot_bgr,
                        best_obs.bbox_px,
                        voted_label,
                        st.track_id,
                        best_obs.conf,
                        max_dim=640
                    )
                else:
                    snapshot = padded_crop

                rebuilt_summaries.append(TrackSummary(
                    track_id=st.track_id,
                    label=voted_label,
                    t_start=first_obs.iso_time,
                    t_end=last_obs.iso_time,
                    t_best=best_obs.iso_time,
                    offset_start=first_obs.offset_s,
                    offset_end=last_obs.offset_s,
                    offset_best=best_obs.offset_s,
                    bbox_px=best_obs.bbox_px,
                    bbox_norm=best_obs.bbox_norm,
                    best_crop=padded_crop,
                    best_snapshot_bgr=snapshot,
                    group=group,
                    colors=colors_json,
                    quality=round(best_q, 4),
                    hits=st.hits
                ))
            summaries = rebuilt_summaries

    return summaries, sampled_frames, dropped_counts, merge_logs
