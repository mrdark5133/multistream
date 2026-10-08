import cv2
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector
from src.ingest.detect_track import load_yolo_world

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

def main():
    print("=" * 80)
    print("REPRODUCING EXPERIMENT 1: SETUP A vs SETUP B AT STRIDE 5")
    print("=" * 80)

    # Setup A: YOLO-World alone
    yolo_world, world_classes = load_yolo_world(
        weights_path="yolov8s-worldv2.pt",
        vocab_path="config/vocab.yaml",
        device="cuda:0"
    )

    # Setup B: Hybrid Detector (YOLO11s on COCO keep classes + YOLO-World on custom vocab)
    hybrid = HybridDetector(
        coco_weights="yolo11s.pt",
        world_weights="yolov8s-worldv2.pt",
        vocab_path="config/vocab.yaml",
        device="cuda:0"
    )

    setup_a_counts = Counter()
    setup_b_counts = Counter()
    setup_a_per_clip = {}
    setup_b_per_clip = {}

    for clip_path in DEV_CLIPS:
        cap = cv2.VideoCapture(clip_path)
        frame_idx = 0
        clip_a_counter = Counter()
        clip_b_counter = Counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:  # Stride 5
                # Setup A
                res_a = yolo_world.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
                if res_a.boxes is not None:
                    for c in res_a.boxes.cls.tolist():
                        cls_idx = int(c)
                        lbl = world_classes[cls_idx] if cls_idx < len(world_classes) else f"cls_{cls_idx}"
                        clip_a_counter[lbl] += 1
                        setup_a_counts[lbl] += 1

                # Setup B
                dets_b = hybrid.detect(frame, imgsz=640)
                for d in dets_b:
                    lbl = d["label"]
                    clip_b_counter[lbl] += 1
                    setup_b_counts[lbl] += 1

            frame_idx += 1
        cap.release()
        setup_a_per_clip[clip_path] = clip_a_counter
        setup_b_per_clip[clip_path] = clip_b_counter

    print("\n--- Setup A: YOLO-World Alone Detections Per Class (Stride 5) ---")
    sum_a = sum(setup_a_counts.values())
    for lbl, count in sorted(setup_a_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:22s}: {count:4d}")
    print(f"Total Setup A: {sum_a}")

    print("\n--- Setup B: Hybrid YOLO11s + YOLO-World Detections Per Class (Stride 5) ---")
    sum_b = sum(setup_b_counts.values())
    for lbl, count in sorted(setup_b_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:22s}: {count:4d}")
    print(f"Total Setup B: {sum_b}")

    print("\n--- Per-Clip Breakdown ---")
    for clip in DEV_CLIPS:
        print(f"\nClip: {clip}")
        print("  Setup A:", dict(setup_a_per_clip[clip]))
        print("  Setup B:", dict(setup_b_per_clip[clip]))

if __name__ == "__main__":
    main()
