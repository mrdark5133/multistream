"""
Phase 0 Fix-Up 3: Comprehensive Benchmarks on upright clips:
test_video01.mp4, test_video02.mp4, test_video03.mp4.
"""
import os
import gc
import sys
import time
import subprocess
import yaml
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import torch
import numpy as np
from ultralytics import YOLOWorld
from transformers import AutoProcessor, AutoModel

def get_smi():
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if res.returncode == 0:
        parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return parts[0], parts[1], parts[2]
    return -1.0, -1.0, -1.0

def compute_roc_auc(scores: list[float], binary_labels: list[int]) -> float:
    n_pos = sum(binary_labels)
    n_neg = len(binary_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[idx] for idx, lbl in enumerate(binary_labels) if lbl == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def compute_precision_recall(ranked_indices: list[int], binary_labels: list[int], k: int):
    total_pos = sum(binary_labels)
    top_k = ranked_indices[:k]
    hits = sum(binary_labels[idx] for idx in top_k)
    recall = hits / total_pos if total_pos > 0 else 0.0
    precision = hits / k if k > 0 else 0.0
    return precision, recall

def main():
    print("=" * 80)
    print("PHASE 0 FIX-UP 3: COMPREHENSIVE BENCHMARK ON NEW UPRIGHT FOOTAGE")
    print("=" * 80)

    # 1. Standalone Latency Benchmark on Upright Frame (test_video01 frame at 1.0s)
    print("\n--- 1. Standalone YOLO-World Latency Benchmark (Upright Frame) ---")
    frame_ref_path = "footage/orientation_check/frame_test_video01.jpg"
    yolo = YOLOWorld("yolov8s-worldv2.pt")
    vocab_classes = yaml.safe_load(open("config/vocab.yaml", "r", encoding="utf-8"))["classes"]
    yolo.set_classes(vocab_classes)
    yolo.to("cuda:0")

    # 3 warmup
    for _ in range(3):
        _ = yolo.predict(frame_ref_path, device="cuda:0", verbose=False)

    latencies_ms = []
    for i in range(1, 21):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = yolo.predict(frame_ref_path, device="cuda:0", verbose=False)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000
        latencies_ms.append(lat)

    import statistics
    print(f"  Warmup: 3 passes complete")
    print(f"  Timed Runs (N=20): Median={statistics.median(latencies_ms):.2f} ms | Mean={statistics.mean(latencies_ms):.2f} ms | Min={min(latencies_ms):.2f} ms | Max={max(latencies_ms):.2f} ms")
    print(f"  Detections on upright frame: {len(res[0].boxes)}")

    # 2. Extract Real Crops across all three clips
    print("\n--- 2. Crop Extraction across test_video01, test_video02, test_video03 ---")
    crops_dir = Path("footage/new_eval_crops/crops")
    crops_dir.mkdir(parents=True, exist_ok=True)

    extracted_crops = []
    crop_id = 0

    # Sampling points across the 3 videos
    video_samples = [
        ("footage/test_video01.mp4", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
        ("footage/test_video02.mp4", [1.0, 3.0, 5.0, 7.0]),
        ("footage/test_video03.mp4", [0.33, 0.67, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 14.5])
    ]

    for vid_path, times in video_samples:
        cap = cv2.VideoCapture(vid_path)
        vid_stem = Path(vid_path).stem
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pred = yolo.predict(frame, device="cuda:0", verbose=False)
            boxes = pred[0].boxes
            for b in boxes:
                cls_id = int(b.cls[0].item())
                label = vocab_classes[cls_id] if cls_id < len(vocab_classes) else f"class_{cls_id}"
                conf = float(b.conf[0].item())
                xyxy = [int(x) for x in b.xyxy[0].tolist()]
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if w < 25 or h < 25 or conf < 0.25:
                    continue
                # Map category
                if "person" in label:
                    category = "person"
                elif "bag" in label or "backpack" in label or "handbag" in label:
                    category = "bag"
                elif "hat" in label or "cap" in label or "helmet" in label:
                    category = "hat"
                else:
                    category = "other"

                crop = pil_frame.crop(xyxy)
                crop_path = crops_dir / f"crop_{crop_id:02d}_{category}_{vid_stem}_{t:.1f}s.jpg"
                crop.save(crop_path)
                extracted_crops.append({
                    "crop_id": crop_id,
                    "video": vid_stem,
                    "timestamp": t,
                    "raw_label": label,
                    "category": category,
                    "conf": conf,
                    "bbox": xyxy,
                    "image": crop,
                    "filepath": str(crop_path)
                })
                crop_id += 1
        cap.release()

    print(f"  Total valid crops extracted: {len(extracted_crops)}")
    cat_counts = {}
    for c in extracted_crops:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print(f"  Category breakdown: {cat_counts}")

    # Build Contact Sheet
    contact_path = Path("footage/crops_contact_sheet.jpg")
    cols = 6
    rows = (len(extracted_crops) + cols - 1) // cols
    tile_w, tile_h, pad = 200, 240, 10
    total_w = cols * tile_w + (cols + 1) * pad
    total_h = rows * tile_h + (rows + 1) * pad + 50
    sheet = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 15), f"MULTIStream Phase 0 Crops Contact Sheet (Upright Videos 01-03, N={len(extracted_crops)})", fill=(240, 240, 240))

    for i, c in enumerate(extracted_crops):
        r, col = i // cols, i % cols
        x0 = pad + col * (tile_w + pad)
        y0 = 60 + pad + r * (tile_h + pad)
        draw.rectangle([x0, y0, x0 + tile_w, y0 + tile_h], fill=(40, 40, 45), outline=(70, 70, 80))
        img_copy = c["image"].copy()
        img_copy.thumbnail((tile_w - 16, tile_h - 70))
        iw, ih = img_copy.size
        sheet.paste(img_copy, (x0 + (tile_w - iw) // 2, y0 + 8))
        draw.text((x0 + 8, y0 + tile_h - 45), f"ID:{c['crop_id']:02d} [{c['category'].upper()}]", fill=(0, 220, 180))
        draw.text((x0 + 8, y0 + tile_h - 25), f"{c['raw_label']} ({c['video']}@{c['timestamp']}s)", fill=(200, 200, 200))
    sheet.save(contact_path, quality=95)
    print(f"  Contact sheet saved to: {contact_path}")

    # 3. Load SigLIP-Base on GPU
    print("\n--- 3. Semantic Similarity & Retrieval on Upright Crops (SigLIP-Base) ---")
    model_id = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    siglip = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
    siglip.eval()

    # Pre-extract crop embeddings
    crop_embs = []
    with torch.no_grad():
        for c in extracted_crops:
            inp = processor(images=c["image"], return_tensors="pt").to("cuda:0")
            if "pixel_values" in inp:
                inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
            feat = siglip.get_image_features(**inp)
            emb = feat.pooler_output if hasattr(feat, "pooler_output") else feat
            if isinstance(emb, tuple): emb = emb[0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            crop_embs.append(emb)
    crop_embs = torch.cat(crop_embs, dim=0)

    # 4 Prompt variants with identical 7-candidate list
    variants = {
        "variant_1_bare": {
            "candidates": ["a person", "a bag", "a hat", "a motor vehicle or car", "a dog or pet animal", "a tree or plant", "an outdoor gate or entrance"],
            "target_map": {"person": "a person", "bag": "a bag", "hat": "a hat", "other": "a tree or plant"}
        },
        "variant_2_photo_of": {
            "candidates": ["a photo of a person", "a photo of a bag", "a photo of a hat", "a photo of a motor vehicle or car", "a photo of a dog or pet animal", "a photo of a tree or plant", "a photo of an outdoor gate or entrance"],
            "target_map": {"person": "a photo of a person", "bag": "a photo of a bag", "hat": "a photo of a hat", "other": "a photo of a tree or plant"}
        },
        "variant_3_descriptive": {
            "candidates": ["a pedestrian walking", "a tote bag or handbag", "a hat or cap worn on head", "a motor vehicle or automobile", "a domestic dog or canine", "a leafy tree or plant", "a security gate or entrance barrier"],
            "target_map": {"person": "a pedestrian walking", "bag": "a tote bag or handbag", "hat": "a hat or cap worn on head", "other": "a leafy tree or plant"}
        },
        "variant_4_composite": {
            "candidates": ["a walking person", "a carried handbag or tote bag", "a headwear hat or cap", "a motor vehicle on a road", "a pet dog on a leash", "a roadside tree or bush", "a metal gate or barrier"],
            "target_map": {"person": "a walking person", "bag": "a carried handbag or tote bag", "hat": "a headwear hat or cap", "other": "a roadside tree or bush"}
        }
    }

    variant_results = {}
    for vname, vdata in variants.items():
        cands = vdata["candidates"]
        tmap = vdata["target_map"]
        with torch.no_grad():
            t_inp = processor(text=cands, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda:0")
            t_out = siglip.get_text_features(**t_inp)
            t_emb = t_out.pooler_output if hasattr(t_out, "pooler_output") else t_out
            if isinstance(t_emb, tuple): t_emb = t_emb[0]
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

        sims = (crop_embs @ t_emb.T).cpu().numpy()
        top1_correct = 0
        per_class_correct = {cat: 0 for cat in cat_counts}

        for idx, c in enumerate(extracted_crops):
            target = tmap[c["category"]]
            sim_scores = sims[idx]
            ranked = sorted(zip(cands, sim_scores), key=lambda x: x[1], reverse=True)
            if ranked[0][0] == target:
                top1_correct += 1
                per_class_correct[c["category"]] += 1

        total = len(extracted_crops)
        acc = top1_correct / total * 100
        variant_results[vname] = {
            "accuracy": acc,
            "correct": top1_correct,
            "total": total,
            "per_class": {cat: (per_class_correct[cat], cat_counts[cat], per_class_correct[cat] / cat_counts[cat] * 100 if cat_counts[cat] > 0 else 0) for cat in cat_counts}
        }
        print(f"  {vname:24s}: {top1_correct}/{total} top-1 ({acc:.1f}%) | Per-class: {variant_results[vname]['per_class']}")

    # Retrieval-style Test
    print("\n--- 4. Retrieval-Style Test (Precision@k, Recall@k, Chance Level, ROC-AUC) ---")
    retrieval_concepts = [
        {"concept": "person", "query": "a person"},
        {"concept": "bag",    "query": "a bag"},
        {"concept": "hat",    "query": "a hat"}
    ]

    retrieval_metrics = {}
    for r_item in retrieval_concepts:
        concept = r_item["concept"]
        query_text = r_item["query"]
        with torch.no_grad():
            q_inp = processor(text=[query_text], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda:0")
            q_out = siglip.get_text_features(**q_inp)
            q_emb = q_out.pooler_output if hasattr(q_out, "pooler_output") else q_out
            if isinstance(q_emb, tuple): q_emb = q_emb[0]
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

        scores = (crop_embs @ q_emb.T).squeeze(-1).cpu().numpy().tolist()
        binary_labels = [1 if c["category"] == concept else 0 for c in extracted_crops]
        n_pos = sum(binary_labels)
        chance_level = n_pos / len(extracted_crops)
        auc = compute_roc_auc(scores, binary_labels)

        ranked_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        p1, r1 = compute_precision_recall(ranked_indices, binary_labels, 1)
        p3, r3 = compute_precision_recall(ranked_indices, binary_labels, 3)
        p5, r5 = compute_precision_recall(ranked_indices, binary_labels, 5)
        p10, r10 = compute_precision_recall(ranked_indices, binary_labels, 10)

        retrieval_metrics[concept] = {
            "query": query_text,
            "N_true": n_pos,
            "N_total": len(extracted_crops),
            "chance_level": chance_level,
            "auc": auc,
            "p1": p1, "r1": r1,
            "p3": p3, "r3": r3,
            "p5": p5, "r5": r5,
            "p10": p10, "r10": r10
        }
        print(f"  Concept: {concept.upper():6s} (N={n_pos}/{len(extracted_crops)}, Chance Level={chance_level*100:.1f}%) | ROC-AUC={auc:.4f}")
        print(f"    Top-1 : Precision={p1*100:5.1f}% | Recall={r1*100:5.1f}%")
        print(f"    Top-3 : Precision={p3*100:5.1f}% | Recall={r3*100:5.1f}%")
        print(f"    Top-5 : Precision={p5*100:5.1f}% | Recall={r5*100:5.1f}%")
        print(f"    Top-10: Precision={p10*100:5.1f}% | Recall={r10*100:5.1f}%")

    # 5. Orientation Ablation: Upright vs Rotated 180 Degrees
    print("\n--- 5. Orientation Ablation: Upright vs 180° Inverted ---")
    ablation_frames = [
        ("test_video01_1s", "footage/orientation_check/frame_test_video01.jpg"),
        ("test_video02_1s", "footage/orientation_check/frame_test_video02.jpg"),
        ("test_video03_1s", "footage/orientation_check/frame_test_video03.jpg")
    ]

    total_upright_dets = 0
    total_rot180_dets = 0

    for name, fpath in ablation_frames:
        img_up = cv2.imread(fpath)
        img_rot = cv2.rotate(img_up, cv2.ROTATE_180)

        res_up = yolo.predict(img_up, device="cuda:0", verbose=False)
        res_rot = yolo.predict(img_rot, device="cuda:0", verbose=False)

        n_up = len(res_up[0].boxes)
        n_rot = len(res_rot[0].boxes)
        total_upright_dets += n_up
        total_rot180_dets += n_rot

        print(f"  {name:16s}: Upright Detections = {n_up:2d} | Rotated 180° Detections = {n_rot:2d}")

    print(f"  Total Across Frames: Upright={total_upright_dets} detections vs Rotated 180°={total_rot180_dets} detections")
    drop_pct = (1.0 - total_rot180_dets / total_upright_dets) * 100 if total_upright_dets > 0 else 0
    print(f"  Orientation Impact: Inversion causes a {drop_pct:.1f}% drop in object detections.")

    # 6. Co-load VRAM Profiling on Upright Frame (Base and SO400M)
    print("\n--- 6. Co-load VRAM Profiling on Upright Frame ---")
    del yolo, siglip
    gc.collect()
    torch.cuda.empty_cache()

    for m_id, name in [("google/siglip-base-patch16-224", "SigLIP-Base"), ("google/siglip-so400m-patch14-384", "SigLIP-SO400M")]:
        u_init, f_init, _ = get_smi()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Load YOLO
        y_mod = YOLOWorld("yolov8s-worldv2.pt")
        y_mod.set_classes(vocab_classes)
        _ = y_mod.predict(frame_ref_path, device="cuda:0", verbose=False)
        
        # Load SigLIP
        proc = AutoProcessor.from_pretrained(m_id)
        sig_mod = AutoModel.from_pretrained(m_id, torch_dtype=torch.float16).to("cuda:0")
        sig_mod.eval()
        
        # Predict
        pil_img = Image.open(frame_ref_path).convert("RGB")
        inps = proc(text=["a person", "a car", "a street"], images=pil_img, padding="max_length", max_length=64, return_tensors="pt").to("cuda:0")
        if "pixel_values" in inps: inps["pixel_values"] = inps["pixel_values"].to(torch.float16)
        with torch.no_grad():
            _ = sig_mod(**inps)
            
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        u_fin, f_fin, _ = get_smi()
        print(f"  {name:15s}: Init Used={u_init:.1f} MiB | Final Used={u_fin:.1f} MiB | Final Free={f_fin:.1f} MiB | Peak Alloc={peak_alloc:.1f} MB | Delta={u_fin - u_init:+.1f} MiB")
        del y_mod, sig_mod, proc, inps, pil_img
        gc.collect()
        torch.cuda.empty_cache()

    # 7. Allocation Attribution Breakdown per Step
    print("\n--- 7. Allocation Attribution Breakdown per Step ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_used, base_free, _ = get_smi()
    print(f"  Baseline (Idle Desktop)            : nvidia-smi used = {base_used:.1f} MiB")

    # CUDA Context init
    x = torch.empty(1, device="cuda:0")
    smi_ctx, _, _ = get_smi()
    print(f"  Step 1: PyTorch CUDA Context Init  : nvidia-smi used = {smi_ctx:.1f} MiB (Delta: +{smi_ctx - base_used:.1f} MiB) | Torch Alloc: {torch.cuda.memory_allocated()/(1024**2):.2f} MB")

    # YOLO Detector
    m_yolo = YOLOWorld("yolov8s-worldv2.pt")
    m_yolo.to("cuda:0")
    smi_yolo, _, _ = get_smi()
    print(f"  Step 2: YOLO Detector Weights      : nvidia-smi used = {smi_yolo:.1f} MiB (Delta: +{smi_yolo - smi_ctx:.1f} MiB) | Torch Alloc: {torch.cuda.memory_allocated()/(1024**2):.2f} MB")

    # set_classes
    m_yolo.set_classes(vocab_classes)
    smi_cls, _, _ = get_smi()
    print(f"  Step 3: YOLO set_classes (CLIP)    : nvidia-smi used = {smi_cls:.1f} MiB (Delta: +{smi_cls - smi_yolo:.1f} MiB) | Torch Alloc: {torch.cuda.memory_allocated()/(1024**2):.2f} MB")

    # YOLO Predict
    _ = m_yolo.predict(frame_ref_path, device="cuda:0", verbose=False)
    smi_pred, _, _ = get_smi()
    print(f"  Step 4: YOLO Inference Buffers     : nvidia-smi used = {smi_pred:.1f} MiB (Delta: +{smi_pred - smi_cls:.1f} MiB) | Torch Alloc: {torch.cuda.memory_allocated()/(1024**2):.2f} MB")

    # SigLIP-Base Load
    sig_b = AutoModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float16).to("cuda:0")
    smi_sigb, _, _ = get_smi()
    print(f"  Step 5: SigLIP-Base Weights        : nvidia-smi used = {smi_sigb:.1f} MiB (Delta: +{smi_sigb - smi_pred:.1f} MiB) | Torch Alloc: {torch.cuda.memory_allocated()/(1024**2):.2f} MB")

    # 7. Mem_get_info Ballast Tests
    print("\n--- 7. torch.cuda.mem_get_info Ballast Tests ---")
    ballast_blocks = []
    print(f"  {'Step':8s} | {'Ballast Added':14s} | {'mem_get_info Free':18s} | {'Torch Allocated':16s} | {'nvidia-smi Used':16s}")
    print("  " + "-" * 82)
    for step in range(1, 6):
        # allocate 200 MB ballast
        t = torch.empty(200 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda:0")
        ballast_blocks.append(t)
        free_b, tot_b = torch.cuda.mem_get_info(0)
        alloc_b = torch.cuda.memory_allocated(0)
        smi_u, smi_f, _ = get_smi()
        print(f"  Step {step:2d}  | {step * 200:6d} MB      | {free_b/(1024**2):10.1f} MB        | {alloc_b/(1024**2):10.1f} MB       | {smi_u:10.1f} MiB")

    del ballast_blocks, m_yolo, sig_b
    gc.collect()
    torch.cuda.empty_cache()
    print("\nBenchmark and diagnostics run complete.")

if __name__ == "__main__":
    main()
