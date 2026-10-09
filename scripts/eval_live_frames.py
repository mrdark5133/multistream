import os
import sys
import time
import yaml
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

LIVE_FRAMES_DIR = ROOT_DIR / "footage" / "live_frames"
VOCAB_PATH = ROOT_DIR / "config" / "live_vocab.yaml"

def run_eval():
    if not LIVE_FRAMES_DIR.exists():
        print(f"Error: {LIVE_FRAMES_DIR} directory does not exist.")
        return

    frames = sorted([f for f in LIVE_FRAMES_DIR.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not frames:
        print(f"No frames found in {LIVE_FRAMES_DIR}. Please place the 5 frames in footage/live_frames/.")
        return

    print(f"Found {len(frames)} frames in {LIVE_FRAMES_DIR}: {[f.name for f in frames]}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = yaml.safe_load(f)["classes"]

    # 1. Setup A: Current Live Config (yolov8s-worldv2 with config/vocab.yaml at imgsz 640)
    print("\n" + "=" * 90)
    print("SETTING A: Current Live Config (yolov8s-worldv2, config/vocab.yaml, imgsz=640)")
    print("=" * 90)
    with open(ROOT_DIR / "config" / "vocab.yaml", "r", encoding="utf-8") as f:
        surveillance_vocab = yaml.safe_load(f)["classes"]
    
    yolo_a = YOLO("yolov8s-worldv2.pt")
    yolo_a.set_classes(surveillance_vocab)
    if torch.cuda.is_available():
        yolo_a.to("cuda:0")
    yolo_a.float()

    for p in frames:
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        res = yolo_a.predict(img, conf=0.15, imgsz=640, verbose=False)
        lat = (time.perf_counter() - t0) * 1000.0
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes) > 0:
            for c, conf, box in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy().astype(float), boxes.xyxy.cpu().numpy().astype(int)):
                lbl = surveillance_vocab[c] if c < len(surveillance_vocab) else "object"
                dets.append(f"{lbl} ({conf:.2f}) [box: {box.tolist()}]")
        print(f"Frame {p.name:<20} | Latency: {lat:6.1f} ms | Detections ({len(dets)}): {', '.join(dets) if dets else 'None'}")

    del yolo_a
    torch.cuda.empty_cache()

    # 2. Setting B1: YOLO-World with config/live_vocab.yaml at imgsz 640
    print("\n" + "=" * 90)
    print("SETTING B1: YOLO-World Only (config/live_vocab.yaml, imgsz=640)")
    print("=" * 90)
    yolo_b1 = YOLO("yolov8s-worldv2.pt")
    yolo_b1.set_classes(vocab)
    if torch.cuda.is_available():
        yolo_b1.to("cuda:0")
    yolo_b1.float()

    for p in frames:
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        res = yolo_b1.predict(img, conf=0.15, imgsz=640, verbose=False)
        lat = (time.perf_counter() - t0) * 1000.0
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes) > 0:
            for c, conf, box in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy().astype(float), boxes.xyxy.cpu().numpy().astype(int)):
                lbl = vocab[c] if c < len(vocab) else "object"
                dets.append(f"{lbl} ({conf:.2f}) [box: {box.tolist()}]")
        print(f"Frame {p.name:<20} | Latency: {lat:6.1f} ms | Detections ({len(dets)}): {', '.join(dets) if dets else 'None'}")

    # 3. Setting B2: YOLO-World with config/live_vocab.yaml at imgsz 960
    print("\n" + "=" * 90)
    print("SETTING B2: YOLO-World Only (config/live_vocab.yaml, imgsz=960)")
    print("=" * 90)
    for p in frames:
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        res = yolo_b1.predict(img, conf=0.15, imgsz=960, verbose=False)
        lat = (time.perf_counter() - t0) * 1000.0
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes) > 0:
            for c, conf, box in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy().astype(float), boxes.xyxy.cpu().numpy().astype(int)):
                lbl = vocab[c] if c < len(vocab) else "object"
                dets.append(f"{lbl} ({conf:.2f}) [box: {box.tolist()}]")
        print(f"Frame {p.name:<20} | Latency: {lat:6.1f} ms | Detections ({len(dets)}): {', '.join(dets) if dets else 'None'}")

if __name__ == "__main__":
    run_eval()
