import cv2
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector, get_object_group, box_iou
from src.ingest.track_engine import (
    create_tracker,
    run_tracker_on_detections,
    compute_crop_quality,
    filter_tracks,
    stitch_tracks,
    pad_to_square,
    TrackState,
    TrackObservation as EngineTrackObs
)
from src.ingest.embed import SigLIPEmbedder

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

HAND_COUNTED_REAL_OBJECTS = {
    "footage/test_video01.mp4": 1,
    "footage/test_video02.mp4": 1,
    "footage/test_video03.mp4": 3,
    "footage/test_landscape.mp4": 8,
    "footage/test_landscape2.mp4": 6
}

def apply_class_group_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.60) -> List[Dict[str, Any]]:
    """Group detections by group (person, vehicle, object) and suppress boxes with IoU >= iou_thresh."""
    if len(detections) <= 1:
        return detections

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for d in detections:
        grp = d.get("group", get_object_group(d["label"]))
        by_group.setdefault(grp, []).append(d)

    kept: List[Dict[str, Any]] = []
    for grp, group_dets in by_group.items():
        # Sort by confidence descending
        group_dets.sort(key=lambda x: x["conf"], reverse=True)
        group_kept: List[Dict[str, Any]] = []
        for d in group_dets:
            b = np.array(d["bbox"])
            suppressed = False
            for kd in group_kept:
                kb = np.array(kd["bbox"])
                if box_iou(b, kb) >= iou_thresh:
                    suppressed = True
                    break
            if not suppressed:
                group_kept.append(d)
        kept.extend(group_kept)
    return kept

def process_clip(clip_path: str, detector: HybridDetector, embedder: SigLIPEmbedder, use_nms: bool):
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = 2

    tracker = create_tracker("bytetrack_tuned")
    raw_tracks: Dict[int, TrackState] = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        offset_s = frame_idx / fps

        if frame_idx % stride == 0:
            dets = detector.detect(frame, imgsz=640)
            if use_nms:
                dets = apply_class_group_nms(dets, iou_thresh=0.60)
            tracked = run_tracker_on_detections(tracker, dets, frame, w, h)

            for td in tracked:
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
                    iso_time=f"2026-10-08T18:00:{offset_s:06.3f}",
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

    filtered, dropped = filter_tracks(raw_tracks, min_duration_s=1.0, min_hits=4, min_mean_conf=0.35)

    # Stitching
    track_padded_crops = []
    for st in filtered:
        best_obs, _ = st.get_best_observation()
        raw_crop = st.best_crop_bgr if st.best_crop_bgr is not None else best_obs.crop_bgr
        track_padded_crops.append(pad_to_square(raw_crop))

    embs = embedder.embed_images(track_padded_crops, initial_batch_size=16) if track_padded_crops else []
    stitched, _, merges = stitch_tracks(filtered, embs, max_gap_s=2.0, max_center_dist=0.25, min_cosine_sim=0.80)

    return len(filtered), len(stitched), len(merges)

def main():
    detector = HybridDetector(device="cuda:0")
    embedder = SigLIPEmbedder(device="cuda:0")

    print("=" * 80)
    print("EVALUATING CLASS-GROUP NMS (IoU 0.6) BEFORE TRACKING")
    print("=" * 80)

    results = []
    for clip in DEV_CLIPS:
        real_n = HAND_COUNTED_REAL_OBJECTS[clip]
        print(f"\nProcessing {clip} (Real Objects = {real_n})...")

        # Before (without pre-tracking class-group NMS)
        filt_before, stitch_before, m_before = process_clip(clip, detector, embedder, use_nms=False)

        # After (with class-group NMS IoU 0.60 before tracking)
        filt_after, stitch_after, m_after = process_clip(clip, detector, embedder, use_nms=True)

        ratio_before = stitch_before / real_n
        ratio_after = stitch_after / real_n

        results.append({
            "clip": clip,
            "real_n": real_n,
            "filt_before": filt_before,
            "stitch_before": stitch_before,
            "ratio_before": round(ratio_before, 2),
            "filt_after": filt_after,
            "stitch_after": stitch_after,
            "ratio_after": round(ratio_after, 2),
        })

    print("\n" + "=" * 80)
    print("TRACKS PER REAL OBJECT: BEFORE vs AFTER CLASS-GROUP NMS (IoU 0.60)")
    print("=" * 80)
    print(f"{'Clip':28s} | {'Real':4s} | {'Stitched (Before)':17s} | {'Trk/Obj (Before)':16s} | {'Stitched (After)':16s} | {'Trk/Obj (After)':15s}")
    print("-" * 105)
    for r in results:
        print(f"{r['clip']:28s} | {r['real_n']:4d} | {r['stitch_before']:17d} | {r['ratio_before']:16.2f} | {r['stitch_after']:16d} | {r['ratio_after']:15.2f}")

if __name__ == "__main__":
    main()
