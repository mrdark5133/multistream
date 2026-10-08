import os
import shutil
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

def main():
    inspect_dir = Path("eval/inspect_crops")
    if inspect_dir.exists():
        shutil.rmtree(inspect_dir)
    inspect_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridDetector(device="cuda:0")

    targets = ["chair", "bicycle", "clock"]
    crops_collected = {t: [] for t in targets}
    max_crops = 10

    print("Collecting crops from HybridDetector at stride 5...")

    for clip_path in DEV_CLIPS:
        stem = Path(clip_path).stem
        cap = cv2.VideoCapture(clip_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 5 == 0:
                dets = detector.detect(frame, imgsz=640)
                for d in dets:
                    lbl = d["label"]
                    if lbl in crops_collected and len(crops_collected[lbl]) < max_crops:
                        bx1, by1, bx2, by2 = d["bbox"]
                        crop = frame[by1:by2, bx1:bx2].copy()
                        if crop.size > 0:
                            crops_collected[lbl].append({
                                "clip": stem,
                                "frame_idx": frame_idx,
                                "label": lbl,
                                "conf": d["conf"],
                                "bbox": d["bbox"],
                                "crop": crop
                            })

            frame_idx += 1
        cap.release()

    print("\nSummary of Collected Crops at Stride 5:")
    saved_paths = []
    for t in targets:
        items = crops_collected[t]
        print(f"Target '{t}': {len(items)} crops collected")
        for idx, it in enumerate(items, 1):
            fname = f"{it['label']}_{it['clip']}_f{it['frame_idx']:04d}_{idx:02d}.jpg"
            fpath = inspect_dir / fname
            cv2.imwrite(str(fpath), it["crop"])
            saved_paths.append((str(fpath), it["label"], it["conf"], it["clip"], it["frame_idx"]))
            print(f"  Saved: {fpath} | Label: {it['label']} | Conf: {it['conf']:.3f}")

    return saved_paths

if __name__ == "__main__":
    main()
