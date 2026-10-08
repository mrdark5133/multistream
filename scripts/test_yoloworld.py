"""
Phase 0 Task 4: Download and load YOLO-World (yolov8s-worldv2.pt) and apply config/vocab.yaml
"""
import os
import sys
import time
import yaml
import torch
from pathlib import Path
from ultralytics import YOLOWorld

def main():
    print("=" * 60)
    print("Phase 0 Task 4: Testing YOLO-World (yolov8s-worldv2.pt)")
    print("=" * 60)

    vocab_path = Path("config/vocab.yaml")
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    classes = data.get("classes", [])
    print(f"Loaded {len(classes)} classes from {vocab_path}")

    model_name = "yolov8s-worldv2.pt"
    existing_file = Path(model_name)
    print(f"Checking existing model file: {existing_file.exists()}")

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Loading {model_name}...")
    model = YOLOWorld(model_name)
    load_time = time.perf_counter() - t0
    
    file_size_bytes = Path(model_name).stat().st_size if Path(model_name).exists() else 0
    print(f"Model file size: {file_size_bytes / (1024**2):.2f} MB ({file_size_bytes} bytes)")
    print(f"Model load time: {load_time:.2f} s")

    print(f"Setting {len(classes)} custom classes...")
    t_set = time.perf_counter()
    model.set_classes(classes)
    set_time = time.perf_counter() - t_set
    print(f"set_classes completed in {set_time:.2f} s")

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak Torch VRAM allocated during YOLO-World load: {peak_vram:.2f} MB")
    print("YOLO-World test PASSED.")

if __name__ == "__main__":
    main()
