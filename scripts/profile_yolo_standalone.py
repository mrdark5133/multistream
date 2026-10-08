"""
Phase 0 Fix-up Task 4: Standalone YOLO-World VRAM and Latency Benchmark
Load YOLO-World alone on GPU, 3 warmup runs, 20 timed inferences on real frame.
"""
import time
import statistics
import subprocess
import yaml
from pathlib import Path
import torch
from ultralytics import YOLOWorld

def get_smi_used():
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if res.returncode == 0:
        return float(res.stdout.strip())
    return -1.0

def main():
    print("=" * 65)
    print("Phase 0 Fix-up Task 4: YOLO-World Standalone Benchmark")
    print("=" * 65)

    image_path = "footage/sample_frame.jpg"
    vocab_path = "config/vocab.yaml"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    smi_before = get_smi_used()

    print(f"Loading YOLO-World (yolov8s-worldv2.pt)...")
    t0 = time.perf_counter()
    model = YOLOWorld("yolov8s-worldv2.pt")
    with open(vocab_path, "r", encoding="utf-8") as f:
        classes = yaml.safe_load(f).get("classes", [])
    model.set_classes(classes)
    load_time = time.perf_counter() - t0
    print(f"Loaded and configured 62 classes in {load_time:.2f} s")

    # Move to GPU explicitly
    model.to("cuda:0")

    # 3 warm-up runs
    print("\nRunning 3 warmup inferences on real frame...")
    for w in range(3):
        _ = model.predict(image_path, device="cuda:0", verbose=False)
    print("Warmup complete.")

    # 20 timed runs
    print("\nRunning 20 timed inferences...")
    latencies_ms = []
    for i in range(1, 21):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        res = model.predict(image_path, device="cuda:0", verbose=False)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t_start) * 1000
        latencies_ms.append(lat)
        print(f"  Run {i:02d}: {lat:6.2f} ms | Detections: {len(res[0].boxes)}")

    # Measurements
    torch_alloc = torch.cuda.memory_allocated() / (1024**2)
    torch_peak = torch.cuda.max_memory_allocated() / (1024**2)
    smi_after = get_smi_used()

    median_lat = statistics.median(latencies_ms)
    worst_lat = max(latencies_ms)
    mean_lat = statistics.mean(latencies_ms)
    min_lat = min(latencies_ms)

    print("\n" + "=" * 50)
    print("STANDALONE YOLO-WORLD RESULTS:")
    print(f"torch.cuda.memory_allocated    : {torch_alloc:.2f} MB")
    print(f"torch.cuda.max_memory_allocated: {torch_peak:.2f} MB")
    print(f"nvidia-smi Used VRAM (initial) : {smi_before:.1f} MiB")
    print(f"nvidia-smi Used VRAM (active)  : {smi_after:.1f} MiB")
    print(f"Net GPU delta from YOLO        : {smi_after - smi_before:+.1f} MiB")
    print(f"Median Inference Latency       : {median_lat:.2f} ms")
    print(f"Worst (Max) Inference Latency  : {worst_lat:.2f} ms")
    print(f"Mean Inference Latency         : {mean_lat:.2f} ms")
    print(f"Min Inference Latency          : {min_lat:.2f} ms")
    print("=" * 50)

if __name__ == "__main__":
    main()
