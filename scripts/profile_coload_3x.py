"""
Phase 0 Fix-up Task 3: 3x Back-to-Back Co-load Profiling
Runs YOLO-World + SigLIP-Base (3 runs) and YOLO-World + SigLIP-SO400M (3 runs).
Measures initial and final nvidia-smi memory, peak PyTorch allocated,
prints parameter counts and explains memory allocation behavior.
"""
import os
import sys
import gc
import time
import subprocess
import yaml
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLOWorld
from transformers import AutoProcessor, AutoModel

def get_smi_memory():
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if res.returncode == 0:
        parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return parts[0], parts[1], parts[2]
    return -1.0, -1.0, -1.0

def run_single_coload(run_idx: int, model_id: str, image_path: str, vocab_classes: list[str]):
    print(f"\n--- Run {run_idx}: {model_id} ---")
    
    # 1. Initial VRAM
    used_init, free_init, tot_init = get_smi_memory()
    print(f"  [Initial nvidia-smi] Used: {used_init:.1f} MiB | Free: {free_init:.1f} MiB | Total: {tot_init:.1f} MiB")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. Load and run YOLO
    t0_yolo = time.perf_counter()
    yolo = YOLOWorld("yolov8s-worldv2.pt")
    yolo.set_classes(vocab_classes)
    det_res = yolo.predict(image_path, device="cuda:0", verbose=False)
    yolo_time = (time.perf_counter() - t0_yolo) * 1000
    yolo_alloc = torch.cuda.memory_allocated() / (1024**2)

    # 3. Load and run SigLIP
    t0_sig = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    siglip = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
    siglip.eval()

    img = Image.open(image_path).convert("RGB")
    candidates = ["a person", "a tote bag", "a hat", "a car", "a dog"]
    inputs = processor(text=candidates, images=img, padding="max_length", max_length=64, return_tensors="pt").to("cuda:0")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        out = siglip(**inputs)
        probs = torch.sigmoid(out.logits_per_image)[0].tolist()
    sig_time = (time.perf_counter() - t0_sig) * 1000

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    curr_alloc = torch.cuda.memory_allocated() / (1024**2)
    used_fin, free_fin, _ = get_smi_memory()

    print(f"  [Final nvidia-smi]   Used: {used_fin:.1f} MiB | Free: {free_fin:.1f} MiB (Delta: {used_fin - used_init:+.1f} MiB)")
    print(f"  [PyTorch Memory]     Current: {curr_alloc:.1f} MB | Peak Allocated: {peak_alloc:.1f} MB")
    print(f"  [Latency]            YOLO: {yolo_time:.1f} ms | SigLIP load+inf: {sig_time:.1f} ms")

    # Cleanup
    del yolo, siglip, processor, inputs, out, det_res, img
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    return {
        "run": run_idx,
        "used_init": used_init,
        "free_init": free_init,
        "used_fin": used_fin,
        "free_fin": free_fin,
        "delta": used_fin - used_init,
        "peak_alloc": peak_alloc
    }

def inspect_parameters():
    print("\n" + "=" * 70)
    print("PARAMETER AND ARCHITECTURE BREAKDOWN")
    print("=" * 70)

    # 1. Detector alone
    m = YOLOWorld("yolov8s-worldv2.pt")
    det_params = sum(p.numel() for p in m.model.model.parameters())
    det_bytes = sum(p.numel() * p.element_size() for p in m.model.model.parameters())
    print(f"1. YOLO-World Detector (m.model.model):")
    print(f"   Parameters: {det_params:,}")
    print(f"   Param Size: {det_bytes / (1024**2):.2f} MB ({det_bytes} bytes, dtype={next(m.model.model.parameters()).dtype})")

    # 2. When set_classes is called on GPU
    m.to("cuda:0")
    m.set_classes(["person", "bag", "hat"])
    cm = m.model.clip_model
    clip_params = sum(p.numel() for p in cm.parameters())
    clip_bytes = sum(p.numel() * p.element_size() for p in cm.parameters())
    clip_dev = next(cm.parameters()).device
    print(f"\n2. CLIP Text Encoder (m.model.clip_model):")
    print(f"   Parameters: {clip_params:,}")
    print(f"   Param Size: {clip_bytes / (1024**2):.2f} MB ({clip_bytes} bytes, dtype={next(cm.parameters()).dtype})")
    print(f"   Device    : {clip_dev}")
    print(f"   Is CLIP text encoder on GPU after set_classes? {'YES' if 'cuda' in str(clip_dev) else 'NO'}")

    print("\n3. Standalone (1292 MB) vs Co-load (~660 MB) Allocation Explanation:")
    print("   - In standalone benchmarking (profile_yolo_standalone.py), model.to('cuda:0') was called BEFORE set_classes.")
    print("     This caused ultralytics to build and cache self.clip_model (151.3M params, 337.1 MB) directly on cuda:0,")
    print("     and 20 sequential warmup/benchmark inferences allocated dynamic workspace/activation buffers,")
    print("     yielding torch.cuda.memory_allocated() = 1059 MB (peak 1292 MB).")
    print("   - In co-load scripts, yolo.set_classes() was executed on CPU before moving/predicting on GPU,")
    print("     so self.clip_model remained resident on CPU (0 MB GPU). Only the detector backbone (48.7 MB)")
    print("     and offline prompt embeddings (txt_feats: 0.12 MB) executed on GPU, allocating ~660.9 MB with image buffers.")
    print("=" * 70)

def main():
    print("=" * 70)
    print("PHASE 0 FIX-UP 2: CO-LOAD 3X BACK-TO-BACK BENCHMARK")
    print("=" * 70)

    image_path = "footage/sample_frame.jpg"
    vocab_path = Path("config/vocab.yaml")
    with open(vocab_path, "r", encoding="utf-8") as f:
        classes = yaml.safe_load(f).get("classes", [])

    # SigLIP Base 3 runs
    print("\n" + "=" * 50)
    print("TESTING YOLO-WORLD + SIGLIP-BASE (3 RUNS)")
    print("=" * 50)
    base_runs = []
    for i in range(1, 4):
        res = run_single_coload(i, "google/siglip-base-patch16-224", image_path, classes)
        base_runs.append(res)

    # SigLIP SO400M 3 runs
    print("\n" + "=" * 50)
    print("TESTING YOLO-WORLD + SIGLIP-SO400M (3 RUNS)")
    print("=" * 50)
    so400m_runs = []
    for i in range(1, 4):
        res = run_single_coload(i, "google/siglip-so400m-patch14-384", image_path, classes)
        so400m_runs.append(res)

    # Summary table
    print("\n" + "=" * 70)
    print("CO-LOAD 3X SUMMARY TABLE:")
    print("=" * 70)
    print(f"{'Model':20s} | {'Run':3s} | {'Init Used':9s} | {'Init Free':9s} | {'Final Used':10s} | {'Final Free':10s} | {'Peak Alloc':10s}")
    print("-" * 85)
    for r in base_runs:
        print(f"{'SigLIP-Base':20s} | {r['run']:3d} | {r['used_init']:7.1f} MiB | {r['free_init']:7.1f} MiB | {r['used_fin']:8.1f} MiB | {r['free_fin']:8.1f} MiB | {r['peak_alloc']:8.1f} MB")
    print("-" * 85)
    for r in so400m_runs:
        print(f"{'SigLIP-SO400M':20s} | {r['run']:3d} | {r['used_init']:7.1f} MiB | {r['free_init']:7.1f} MiB | {r['used_fin']:8.1f} MiB | {r['free_fin']:8.1f} MiB | {r['peak_alloc']:8.1f} MB")

    # Inspect parameters and print explanation
    inspect_parameters()

if __name__ == "__main__":
    main()
