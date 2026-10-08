"""
Phase 0 Task 5 & 6: Co-load YOLO-World and SigLIP on GPU and profile memory with a real image.
"""
import os
import sys
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
        return parts[0], parts[1], parts[2] # used, free, total (MB)
    return -1, -1, -1

def profile_model(image_path: str, model_id: str = "google/siglip-so400m-patch14-384", fp16: bool = True):
    print("=" * 70)
    print(f"PROFILING CO-EXISTENCE: YOLO-World + {model_id} (fp16={fp16})")
    print("=" * 70)

    used0, free0, tot0 = get_smi_memory()
    print(f"[Initial nvidia-smi] Used: {used0:.1f} MB | Free: {free0:.1f} MB | Total: {tot0:.1f} MB")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Load YOLO-World
    print("\n1. Loading YOLO-World (yolov8s-worldv2.pt)...")
    t0 = time.perf_counter()
    yolo = YOLOWorld("yolov8s-worldv2.pt")
    vocab_path = Path("config/vocab.yaml")
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            classes = yaml.safe_load(f).get("classes", [])
        yolo.set_classes(classes)
    yolo_load_time = time.perf_counter() - t0
    yolo_alloc = torch.cuda.memory_allocated() / (1024**2)
    used1, free1, _ = get_smi_memory()
    print(f"YOLO loaded in {yolo_load_time:.2f}s | Torch Alloc: {yolo_alloc:.1f} MB | SMI Free: {free1:.1f} MB")

    # 2. Run detector on real image
    print(f"\n2. Running YOLO inference on {image_path}...")
    t_det0 = time.perf_counter()
    det_results = yolo.predict(image_path, device="cuda:0", verbose=False)
    det_time = time.perf_counter() - t_det0
    num_boxes = len(det_results[0].boxes) if det_results else 0
    print(f"Detection completed in {det_time*1000:.1f}ms | Detections found: {num_boxes}")

    # 3. Load SigLIP
    dtype = torch.float16 if fp16 else torch.float32
    print(f"\n3. Loading {model_id} (dtype={dtype}) on GPU...")
    t_sig0 = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        siglip = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to("cuda:0")
        siglip_load_time = time.perf_counter() - t_sig0
        sig_alloc = torch.cuda.memory_allocated() / (1024**2)
        used2, free2, _ = get_smi_memory()
        print(f"SigLIP loaded in {siglip_load_time:.2f}s | Torch Alloc: {sig_alloc:.1f} MB | SMI Free: {free2:.1f} MB")
    except Exception as e:
        print(f"FAILED TO LOAD {model_id} on GPU: {type(e).__name__}: {e}")
        return False

    # 4. Run real image and text embedding through SigLIP
    print("\n4. Running joint image-text inference...")
    try:
        img = Image.open(image_path).convert("RGB")
        candidate_texts = ["a car", "a vehicle", "a person walking", "a dog", "a tree"]
        inputs = processor(text=candidate_texts, images=img, padding="max_length", return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        if fp16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        t_inf0 = time.perf_counter()
        with torch.no_grad():
            outputs = siglip(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)
        inf_time = time.perf_counter() - t_inf0

        print(f"Joint inference time: {inf_time*1000:.1f}ms")
        print("\nSimilarity Scores:")
        for txt, score in zip(candidate_texts, probs[0].tolist()):
            print(f"  {txt:20}: {score:.4f}")

        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        used_end, free_end, _ = get_smi_memory()
        print("\n" + "=" * 50)
        print("MEMORy PROFILE RESULTS:")
        print(f"Peak PyTorch Allocated : {peak_alloc:.1f} MB")
        print(f"nvidia-smi Used VRAM   : {used_end:.1f} MB")
        print(f"nvidia-smi Free VRAM   : {free_end:.1f} MB")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"INFERENCE FAILED: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_models.py <image_path> [model_id]")
        sys.exit(1)
    img_arg = sys.argv[1]
    mid = sys.argv[2] if len(sys.argv) > 2 else "google/siglip-so400m-patch14-384"
    profile_model(img_arg, mid)
