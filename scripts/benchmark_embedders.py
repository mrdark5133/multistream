import cv2
import time
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detect_track import load_yolo_world, process_video_tracks
from src.utils.vram import get_nvml_vram_info


def extract_crops_from_upright_clips(target_count: int = 32):
    """Extract real crops from upright clips (test_video01-03.mp4)."""
    yolo, classes = load_yolo_world(device="cuda:0")
    clips = [
        ("footage/test_video01.mp4", 4.26),
        ("footage/test_video02.mp4", 9.13),
        ("footage/test_video03.mp4", 16.16)
    ]
    crops = []
    for c_path, dur in clips:
        tracks, _ = process_video_tracks(
            video_path=c_path,
            yolo_model=yolo,
            classes=classes,
            start_time_iso="2026-10-08T10:00:00",
            fps=30.0,
            duration_s=dur,
            stride=5,
            conf_thresh=0.25,
            imgsz=640
        )
        for t in tracks:
            if t.best_crop is not None and t.best_crop.size > 0:
                crops.append(t.best_crop)
        if len(crops) >= target_count:
            break
            
    # Clean up yolo from memory
    del yolo
    torch.cuda.empty_cache()
    print(f"Extracted {len(crops)} crops from upright clips.")
    return crops[:target_count]


def benchmark_model(model_id: str, crops: list, batch_sizes=(1, 8, 16), warmup=3, timed=20):
    print(f"\n=======================================================")
    print(f"Benchmarking Image Tower: {model_id}")
    print(f"=======================================================")
    
    vram_tot, vram_init, vram_free = get_nvml_vram_info()
    print(f"Initial VRAM: Used {vram_init} MiB | Free {vram_free} MiB")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
    model.eval()
    
    vram_tot, vram_loaded, vram_free = get_nvml_vram_info()
    print(f"Loaded VRAM:  Used {vram_loaded} MiB (Delta: +{vram_loaded - vram_init} MiB)")
    
    pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
    
    results = {}
    
    for bs in batch_sizes:
        batch = (pil_crops * ((bs // len(pil_crops)) + 1))[:bs]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda:0", dtype=torch.float16)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                out = model.get_image_features(pixel_values=pixel_values)
                _ = out.pooler_output if hasattr(out, "pooler_output") else out
            torch.cuda.synchronize()
            
        # Timed runs
        latencies = []
        for _ in range(timed):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.get_image_features(pixel_values=pixel_values)
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
                feats = feats / feats.norm(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
        latencies = np.array(latencies)
        mean_ms = np.mean(latencies)
        median_ms = np.median(latencies)
        min_ms = np.min(latencies)
        max_ms = np.max(latencies)
        throughput = (bs / (mean_ms / 1000.0))
        
        results[bs] = {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "throughput_fps": throughput
        }
        print(f"Batch {bs:2d}: Mean {mean_ms:6.2f} ms | Median {median_ms:6.2f} ms | Min {min_ms:6.2f} ms | Max {max_ms:6.2f} ms | Throughput: {throughput:6.1f} crops/s")
        
    vram_tot, vram_peak, vram_free = get_nvml_vram_info()
    print(f"Peak VRAM during inference: Used {vram_peak} MiB")
    
    del model
    del processor
    torch.cuda.empty_cache()
    
    return results, vram_loaded, vram_peak


def main():
    crops = extract_crops_from_upright_clips(target_count=32)
    
    # 1. SigLIP-Base
    base_results, base_load, base_peak = benchmark_model(
        "google/siglip-base-patch16-224", crops, batch_sizes=[1, 8, 16], warmup=3, timed=20
    )
    
    # 2. SigLIP-SO400M
    so400m_results, so400m_load, so400m_peak = benchmark_model(
        "google/siglip-so400m-patch14-384", crops, batch_sizes=[1, 8, 16], warmup=3, timed=20
    )
    
    print("\n" + "=" * 90)
    print("TASK 1.11 SUMMARY BENCHMARK TABLE: IMAGE TOWER ONLY")
    print("=" * 90)
    print(f"{'Model':<32} | {'BS':<3} | {'Mean (ms)':<9} | {'Median (ms)':<11} | {'Min (ms)':<8} | {'Max (ms)':<8} | {'Crops/s':<8}")
    print("-" * 90)
    for bs in [1, 8, 16]:
        r = base_results[bs]
        print(f"{'SigLIP-Base (patch16-224)':<32} | {bs:<3} | {r['mean_ms']:<9.2f} | {r['median_ms']:<11.2f} | {r['min_ms']:<8.2f} | {r['max_ms']:<8.2f} | {r['throughput_fps']:<8.1f}")
    print("-" * 90)
    for bs in [1, 8, 16]:
        r = so400m_results[bs]
        print(f"{'SigLIP-SO400M (patch14-384)':<32} | {bs:<3} | {r['mean_ms']:<9.2f} | {r['median_ms']:<11.2f} | {r['min_ms']:<8.2f} | {r['max_ms']:<8.2f} | {r['throughput_fps']:<8.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
