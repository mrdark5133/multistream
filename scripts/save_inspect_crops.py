import cv2
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector

def main():
    out_dir = Path("eval/inspect_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridDetector(device="cuda:0")

    targets = {"chair": 10, "bicycle": 10, "clock": 10, "truck_video01": 10}
    saved = {"chair": 0, "bicycle": 0, "clock": 0, "truck_video01": 0}

    clips = [
        "footage/test_video01.mp4",
        "footage/test_video02.mp4",
        "footage/test_video03.mp4",
        "footage/test_landscape.mp4",
        "footage/test_landscape2.mp4"
    ]

    for cp in clips:
        cap = cv2.VideoCapture(cp)
        f_idx = 0
        is_video01 = ("test_video01" in cp)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % 2 == 0:
                dets = detector.detect(frame, imgsz=640)
                for d in dets:
                    lbl = d["label"]
                    bx1, by1, bx2, by2 = d["bbox"]
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size == 0:
                        continue

                    if is_video01 and lbl == "truck" and saved["truck_video01"] < 10:
                        saved["truck_video01"] += 1
                        idx_num = saved["truck_video01"]
                        cv2.imwrite(str(out_dir / f"truck_video01_{idx_num:02d}.jpg"), crop)

                    if lbl in ("chair", "bicycle", "clock") and saved[lbl] < 10:
                        saved[lbl] += 1
                        idx_num = saved[lbl]
                        cv2.imwrite(str(out_dir / f"{lbl}_{idx_num:02d}.jpg"), crop)
            f_idx += 1
        cap.release()

    print("Saved inspection crops:")
    for k, v in saved.items():
        print(f"  {k}: {v} crops saved to {out_dir}")

if __name__ == "__main__":
    main()
