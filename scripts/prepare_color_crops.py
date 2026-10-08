import cv2
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector, get_object_group
from src.utils.lab_color import extract_track_colors
from src.ingest.track_engine import pad_to_square
from src.ingest.embed import SigLIPEmbedder
import numpy as np

def main():
    out_dir = Path("eval/color_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridDetector(device="cuda:0")
    embedder = SigLIPEmbedder(device="cuda:0")

    COLOR_CANDIDATES = ["red", "blue", "green", "yellow", "black", "white", "silver", "grey", "orange", "brown"]
    color_prompts = [f"a photo of a {c} object" for c in COLOR_CANDIDATES]
    color_text_embs = embedder.embed_text(color_prompts)

    clips_to_sample = [
        ("footage/test_landscape.mp4", ["car", "bus", "truck"], 10),      # 10 Landscape cars/buses
        ("footage/test_landscape2.mp4", ["car", "person", "truck"], 10),  # 10 Landscape2 cars/persons
        ("footage/test_video01.mp4", ["person", "truck"], 5),              # 5 Video01
        ("footage/test_video03.mp4", ["motorcycle", "person", "car"], 5)  # 5 Video03
    ]

    selected_crops = [] # list of (crop_img, label, clip_path, frame_idx)

    for clip_path, target_labels, quota in clips_to_sample:
        cap = cv2.VideoCapture(clip_path)
        f_idx = 0
        clip_taken = 0
        while cap.isOpened() and clip_taken < quota:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % 6 == 0:
                dets = detector.detect(frame, imgsz=640)
                for d in dets:
                    lbl = d["label"]
                    if lbl in target_labels:
                        bx1, by1, bx2, by2 = d["bbox"]
                        bw = bx2 - bx1
                        bh = by2 - by1
                        if bw >= 25 and bh >= 25:
                            crop_img = frame[by1:by2, bx1:bx2].copy()
                            selected_crops.append((crop_img, lbl, clip_path, f_idx))
                            clip_taken += 1
                            if clip_taken >= quota:
                                break
            f_idx += 1
        cap.release()

    selected_crops = selected_crops[:30]
    print(f"Collected {len(selected_crops)} crops across clips.")

    # Save images and prepare CSV
    csv_rows = []
    for idx, (crop_img, lbl, clip_path, f_idx) in enumerate(selected_crops, 1):
        filename = f"crop_{idx:02d}.jpg"
        filepath = out_dir / filename
        cv2.imwrite(str(filepath), crop_img)

        group = get_object_group(lbl)
        padded = pad_to_square(crop_img)

        # 1. Lab K-Means top 2
        cdict = extract_track_colors(padded, lbl, group)
        top_lab_colors = []
        for part, clist in cdict.items():
            for item in clist:
                top_lab_colors.append((item["color"], item["fraction"]))
        top_lab_colors.sort(key=lambda x: x[1], reverse=True)
        lab_str = ", ".join([f"{c}:{f:.2f}" for c, f in top_lab_colors[:2]])

        # 2. SigLIP top 1
        c_emb = embedder.embed_images([padded], initial_batch_size=1)[0]
        sims = np.dot(color_text_embs, c_emb)
        siglip_top = COLOR_CANDIDATES[int(np.argmax(sims))]

        # my_color_label is explicitly blank for human hand-labeling
        csv_rows.append({
            "crop_id": f"crop_{idx:02d}",
            "file_path": str(filepath).replace("\\", "/"),
            "clip_source": clip_path,
            "detected_label": lbl,
            "my_color_label": "",
            "lab_kmeans_top2": lab_str,
            "siglip_top1": siglip_top
        })

    csv_path = out_dir / "color_labels_template.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "crop_id", "file_path", "clip_source", "detected_label",
            "my_color_label", "lab_kmeans_top2", "siglip_top1"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved 30 crops to {out_dir}")
    print(f"Generated hand-labeling template: {csv_path}")

if __name__ == "__main__":
    main()
