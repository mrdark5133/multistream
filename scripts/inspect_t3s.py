import cv2
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector, box_iou
from src.ingest.track_engine import create_tracker, run_tracker_on_detections

def main():
    clip = "footage/test_video01.mp4"
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = 2

    detector = HybridDetector(device="cuda:0")
    tracker = create_tracker("bytetrack_tuned")

    target_offset = 3.0
    print(f"fps: {fps}, target_offset: {target_offset}s")

    frame_idx = 0
    active_tracks_at_target = {}
    frame_at_target = None
    target_offset_recorded = None
    target_dets = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        offset_s = frame_idx / fps

        if frame_idx % stride == 0:
            dets = detector.detect(frame, imgsz=640)
            tracked = run_tracker_on_detections(tracker, dets, frame, w, h)

            if abs(offset_s - target_offset) < (1.0 / fps * stride) and frame_at_target is None:
                frame_at_target = frame.copy()
                target_offset_recorded = offset_s
                target_dets = dets
                for td in tracked:
                    active_tracks_at_target[td["track_id"]] = {
                        "bbox": td["bbox"],
                        "label": td["label"],
                        "conf": td["conf"]
                    }

        frame_idx += 1
    cap.release()

    print(f"\nExact recorded offset: {target_offset_recorded:.3f}s")
    print(f"\nDetections feeding into tracker at t={target_offset_recorded:.3f}s:")
    for d in target_dets:
        print(f"  {d['source']:14s} | {d['label']:10s} | conf: {d['conf']:.3f} | bbox: {d['bbox']}")

    print(f"\nAll Active Tracks at t={target_offset_recorded:.3f}s:")
    for tid, tinfo in sorted(active_tracks_at_target.items()):
        print(f"  Track ID {tid:3d} | {tinfo['label']:10s} | conf: {tinfo['conf']:.3f} | bbox: {tinfo['bbox']}")

    person_track_ids = [14, 23, 25, 26, 28, 35]
    active_persons = {tid: active_tracks_at_target[tid] for tid in person_track_ids if tid in active_tracks_at_target}
    print(f"\nTarget Person tracks present at t=3.0s: {list(active_persons.keys())}")

    # Pairwise IoU
    keys = sorted(list(active_persons.keys()))
    print("\nPairwise IoU Matrix of Concurrent Person Tracks at t=3.0s:")
    overlaps_above_half = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            b1 = np.array(active_persons[keys[i]]["bbox"])
            b2 = np.array(active_persons[keys[j]]["bbox"])
            iou = box_iou(b1, b2)
            print(f"  Track {keys[i]:2d} vs Track {keys[j]:2d}: IoU = {iou:.4f} | Box1: {b1.tolist()} | Box2: {b2.tolist()}")
            if iou > 0.5:
                overlaps_above_half.append((keys[i], keys[j], iou))

    # Save annotated frame
    disp = frame_at_target.copy()
    for tid, tinfo in active_tracks_at_target.items():
        bx1, by1, bx2, by2 = tinfo["bbox"]
        color = (0, 0, 255) if tid in person_track_ids else (255, 0, 0)
        cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            disp,
            f"ID:{tid} {tinfo['label']}",
            (bx1, max(20, by1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out_path = "eval/test_video01_t3s_annotated.jpg"
    Path("eval").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, disp)
    print(f"\nSaved annotated frame to: {out_path}")

    print(f"\nHeavy overlaps (> 0.5): {len(overlaps_above_half)}")
    for p in overlaps_above_half:
        print(f"  Tracks ({p[0]}, {p[1]}): IoU = {p[2]:.4f}")

if __name__ == "__main__":
    main()
