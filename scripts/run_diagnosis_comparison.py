import os
import sys
import time
import subprocess
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

VIDEO_PATH = ROOT_DIR / "footage" / "mobile_cam02_20261009_024019.mp4"
OUTPUT_DIR = ROOT_DIR / "footage" / "detect_compare"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_VOCAB_PATH = ROOT_DIR / "config" / "test_vocab.yaml"

with open(TEST_VOCAB_PATH, "r", encoding="utf-8") as f:
    TARGET_CLASSES = yaml.safe_load(f)["classes"]

FRAME_INDICES = [60, 180, 300, 420, 540]


def get_nvidia_smi_vram_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def extract_frames(video_path: Path, frame_indices: List[int]) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
            raw_path = OUTPUT_DIR / f"frame_{idx}_raw.jpg"
            cv2.imwrite(str(raw_path), frame)
    cap.release()
    return frames


def annotate_and_save(img_bgr: np.ndarray, boxes_xyxy: List[List[int]], labels: List[str], confs: List[float], out_path: Path):
    canvas = img_bgr.copy()
    for box, lbl, conf in zip(boxes_xyxy, labels, confs):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{lbl} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), (0, 255, 0), -1)
        cv2.putText(canvas, text, (x1 + 2, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(out_path), canvas)


def run_yoloworld(frames: List[Tuple[int, np.ndarray]], imgsz: int) -> Dict[str, Any]:
    from ultralytics import YOLO

    torch.cuda.empty_cache()
    vram_start = get_nvidia_smi_vram_mib()
    peak_vram = vram_start

    t_load_0 = time.perf_counter()
    yolo = YOLO("yolov8s-worldv2.pt")
    yolo.set_classes(TARGET_CLASSES)
    if torch.cuda.is_available():
        yolo.to("cuda:0")
    yolo.float()
    load_time = time.perf_counter() - t_load_0

    frame_times = []
    class_counts = Counter()
    per_frame_dets = {}

    for idx, frame in frames:
        t0 = time.perf_counter()
        res = yolo.predict(frame, conf=0.15, imgsz=imgsz, verbose=False)
        dt = time.perf_counter() - t0
        frame_times.append(dt)

        cur_vram = get_nvidia_smi_vram_mib()
        if cur_vram > peak_vram:
            peak_vram = cur_vram

        boxes_xyxy = []
        labels = []
        confs = []

        if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
            b = res[0].boxes
            cls_ids = b.cls.cpu().numpy().astype(int)
            b_confs = b.conf.cpu().numpy().astype(float)
            xyxy = b.xyxy.cpu().numpy().astype(int)

            for c, cf, bx in zip(cls_ids, b_confs, xyxy):
                lbl = TARGET_CLASSES[c] if c < len(TARGET_CLASSES) else "object"
                boxes_xyxy.append(bx.tolist())
                labels.append(lbl)
                confs.append(cf)
                class_counts[lbl] += 1

        out_name = f"frame_{idx}_yoloworld_{imgsz}.jpg"
        annotate_and_save(frame, boxes_xyxy, labels, confs, OUTPUT_DIR / out_name)
        per_frame_dets[idx] = list(zip(labels, confs, boxes_xyxy))

    del yolo
    torch.cuda.empty_cache()

    return {
        "model": f"yolov8s-worldv2 (imgsz {imgsz})",
        "avg_sec_per_frame": float(np.mean(frame_times)),
        "peak_vram_mib": peak_vram,
        "class_counts": class_counts,
        "per_frame": per_frame_dets
    }


def run_grounding_dino(frames: List[Tuple[int, np.ndarray]]) -> Dict[str, Any]:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    torch.cuda.empty_cache()
    vram_start = get_nvidia_smi_vram_mib()
    peak_vram = vram_start

    t_load_0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda:0")
    load_time = time.perf_counter() - t_load_0

    gd_text = ". ".join(TARGET_CLASSES) + "."

    frame_times = []
    class_counts = Counter()
    per_frame_dets = {}

    for idx, frame_bgr in frames:
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        t0 = time.perf_counter()
        inputs = proc(images=pil_img, text=gd_text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(H, W)],
            text_labels=[TARGET_CLASSES]
        )[0]
        dt = time.perf_counter() - t0
        frame_times.append(dt)

        cur_vram = get_nvidia_smi_vram_mib()
        if cur_vram > peak_vram:
            peak_vram = cur_vram

        boxes_xyxy = []
        labels = []
        confs = []

        for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"]):
            clean_lbl = lb.strip().lower()
            # Match to nearest target class
            matched_lbl = clean_lbl
            for tc in TARGET_CLASSES:
                if tc in clean_lbl or clean_lbl in tc:
                    matched_lbl = tc
                    break
            boxes_xyxy.append(bx.int().cpu().numpy().tolist())
            labels.append(matched_lbl)
            confs.append(float(sc))
            class_counts[matched_lbl] += 1

        out_name = f"frame_{idx}_grounding_dino.jpg"
        annotate_and_save(frame_bgr, boxes_xyxy, labels, confs, OUTPUT_DIR / out_name)
        per_frame_dets[idx] = list(zip(labels, confs, boxes_xyxy))

    del model, proc
    torch.cuda.empty_cache()

    return {
        "model": "IDEA-Research/grounding-dino-tiny",
        "avg_sec_per_frame": float(np.mean(frame_times)),
        "peak_vram_mib": peak_vram,
        "class_counts": class_counts,
        "per_frame": per_frame_dets
    }


def run_owlv2(frames: List[Tuple[int, np.ndarray]]) -> Dict[str, Any]:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    torch.cuda.empty_cache()
    vram_start = get_nvidia_smi_vram_mib()
    peak_vram = vram_start

    t_load_0 = time.perf_counter()
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda:0")
    load_time = time.perf_counter() - t_load_0

    frame_times = []
    class_counts = Counter()
    per_frame_dets = {}

    for idx, frame_bgr in frames:
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        t0 = time.perf_counter()
        inputs = proc(text=[TARGET_CLASSES], images=pil_img, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=[(H, W)],
            threshold=0.15
        )[0]
        dt = time.perf_counter() - t0
        frame_times.append(dt)

        cur_vram = get_nvidia_smi_vram_mib()
        if cur_vram > peak_vram:
            peak_vram = cur_vram

        boxes_xyxy = []
        labels = []
        confs = []

        for sc, lb_idx, bx in zip(res["scores"], res["labels"], res["boxes"]):
            idx_int = lb_idx.item()
            lbl = TARGET_CLASSES[idx_int] if idx_int < len(TARGET_CLASSES) else "unknown"
            boxes_xyxy.append(bx.int().cpu().numpy().tolist())
            labels.append(lbl)
            confs.append(float(sc))
            class_counts[lbl] += 1

        out_name = f"frame_{idx}_owlv2.jpg"
        annotate_and_save(frame_bgr, boxes_xyxy, labels, confs, OUTPUT_DIR / out_name)
        per_frame_dets[idx] = list(zip(labels, confs, boxes_xyxy))

    del model, proc
    torch.cuda.empty_cache()

    return {
        "model": "google/owlv2-base-patch16-ensemble",
        "avg_sec_per_frame": float(np.mean(frame_times)),
        "peak_vram_mib": peak_vram,
        "class_counts": class_counts,
        "per_frame": per_frame_dets
    }


def main():
    print("=" * 100)
    print("DESK OBJECTS DETECTION DIAGNOSIS")
    print(f"Video: {VIDEO_PATH}")
    print(f"Target Vocabulary ({len(TARGET_CLASSES)} classes): {TARGET_CLASSES}")
    print(f"Frames to sample: {FRAME_INDICES}")
    print("=" * 100)

    frames = extract_frames(VIDEO_PATH, FRAME_INDICES)
    print(f"Extracted {len(frames)} frames to {OUTPUT_DIR}\n")

    results = []

    print("--- Running YOLO-World (imgsz=640) ---")
    res_yw640 = run_yoloworld(frames, imgsz=640)
    results.append(res_yw640)
    print(f"Done: {res_yw640['avg_sec_per_frame']:.3f} s/frame | Peak VRAM: {res_yw640['peak_vram_mib']} MiB\n")

    print("--- Running YOLO-World (imgsz=1280) ---")
    res_yw1280 = run_yoloworld(frames, imgsz=1280)
    results.append(res_yw1280)
    print(f"Done: {res_yw1280['avg_sec_per_frame']:.3f} s/frame | Peak VRAM: {res_yw1280['peak_vram_mib']} MiB\n")

    print("--- Running Grounding DINO Tiny ---")
    res_gd = run_grounding_dino(frames)
    results.append(res_gd)
    print(f"Done: {res_gd['avg_sec_per_frame']:.3f} s/frame | Peak VRAM: {res_gd['peak_vram_mib']} MiB\n")

    print("--- Running OWLv2 Base Patch16 Ensemble ---")
    res_owl = run_owlv2(frames)
    results.append(res_owl)
    print(f"Done: {res_owl['avg_sec_per_frame']:.3f} s/frame | Peak VRAM: {res_owl['peak_vram_mib']} MiB\n")

    # Format Markdown Comparison Table
    print("\n" + "=" * 100)
    print("DIAGNOSIS COMPARISON TABLE (Detection Counts Across 5 Frames, Latency, and Peak VRAM)")
    print("=" * 100)

    header = f"| {'Target Class':<24} | {'YOLO-World (640)':<18} | {'YOLO-World (1280)':<18} | {'Grounding-DINO-Tiny':<20} | {'OWLv2-Base-Ensemble':<20} |"
    sep = f"|{'-'*26}|{'-'*20}|{'-'*20}|{'-'*22}|{'-'*22}|"
    print(header)
    print(sep)

    for cls_name in TARGET_CLASSES:
        c1 = res_yw640["class_counts"].get(cls_name, 0)
        c2 = res_yw1280["class_counts"].get(cls_name, 0)
        c3 = res_gd["class_counts"].get(cls_name, 0)
        c4 = res_owl["class_counts"].get(cls_name, 0)
        print(f"| {cls_name:<24} | {c1:<18} | {c2:<18} | {c3:<20} | {c4:<20} |")

    print(sep)
    # Total detections
    tot1 = sum(res_yw640["class_counts"].values())
    tot2 = sum(res_yw1280["class_counts"].values())
    tot3 = sum(res_gd["class_counts"].values())
    tot4 = sum(res_owl["class_counts"].values())
    print(f"| {'TOTAL DETECTIONS':<24} | {tot1:<18} | {tot2:<18} | {tot3:<20} | {tot4:<20} |")

    # Performance
    print(f"| {'Seconds / Frame':<24} | {res_yw640['avg_sec_per_frame']:<18.3f} | {res_yw1280['avg_sec_per_frame']:<18.3f} | {res_gd['avg_sec_per_frame']:<20.3f} | {res_owl['avg_sec_per_frame']:<20.3f} |")
    print(f"| {'Peak VRAM (nvidia-smi)':<24} | {res_yw640['peak_vram_mib']} MiB{' '*11} | {res_yw1280['peak_vram_mib']} MiB{' '*11} | {res_gd['peak_vram_mib']} MiB{' '*13} | {res_owl['peak_vram_mib']} MiB{' '*13} |")
    print("=" * 100)


if __name__ == "__main__":
    main()
