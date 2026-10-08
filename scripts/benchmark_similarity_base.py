"""
Phase 0 Fix-up Task 1: Comprehensive SigLIP-Base Semantic Similarity Evaluation
Extract >= 10 real crops across multiple timestamps of footage/test_gate.mp4.
Proper SigLIP tokenization (padding="max_length", max_length=64, truncation=True).
Evaluate prompt variants against distractors and compute top-1 accuracy among distractors.
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLOWorld
from transformers import AutoProcessor, AutoModel

def extract_frames_and_crops(video_path: str, timestamps_s: list[float], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    crops_dir = output_dir / "crops"
    frames_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    yolo = YOLOWorld("yolov8s-worldv2.pt")
    vocab_path = Path("config/vocab.yaml")
    with open(vocab_path, "r", encoding="utf-8") as f:
        classes = yaml.safe_load(f).get("classes", [])
    model_classes = classes
    yolo.set_classes(model_classes)
    yolo.to("cuda:0")

    extracted_crops = []
    crop_counter = 0

    for ts in timestamps_s:
        frame_file = frames_dir / f"frame_{ts:.1f}s.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path,
            "-frames:v", "1", "-q:v", "2", str(frame_file)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not frame_file.exists():
            continue

        pil_frame = Image.open(frame_file).convert("RGB")
        results = yolo.predict(frame_file, device="cuda:0", verbose=False)
        boxes = results[0].boxes

        for b_idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            label = model_classes[cls_id] if cls_id < len(model_classes) else f"class_{cls_id}"
            conf = float(box.conf[0].item())
            xyxy = [int(x) for x in box.xyxy[0].tolist()]

            # Filter out tiny/degenerate boxes
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            if w < 20 or h < 20:
                continue

            crop = pil_frame.crop(xyxy)
            crop_path = crops_dir / f"crop_{crop_counter:02d}_{label}_{ts:.1f}s.jpg"
            crop.save(crop_path)

            extracted_crops.append({
                "crop_id": crop_counter,
                "label": label,
                "timestamp_s": ts,
                "conf": conf,
                "bbox": xyxy,
                "crop_path": str(crop_path),
                "image": crop
            })
            crop_counter += 1

    return extracted_crops

def main():
    print("=" * 70)
    print("Phase 0 Fix-up Task 1: SigLIP-Base Semantic Ranking on >=10 Real Crops")
    print("=" * 70)

    video_path = "footage/test_gate.mp4"
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found!")
        sys.exit(1)

    # 1. Extract >= 10 real crops from multiple timestamps
    timestamps = [0.5, 1.5, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    out_dir = Path("footage/eval_crops")
    print(f"Extracting frames at timestamps {timestamps} from {video_path}...")
    crops = extract_frames_and_crops(video_path, timestamps, out_dir)
    print(f"Total valid crops extracted: {len(crops)}")

    if len(crops) < 10:
        print(f"Warning: Only {len(crops)} crops found, need >= 10.")
    else:
        print(f"Successfully collected {len(crops)} real crops (>= 10 requirement met).")

    # 2. Setup SigLIP-Base with proper tokenization (max_length=64, padding="max_length")
    model_id = "google/siglip-base-patch16-224"
    print(f"\nLoading embedder: {model_id} on cuda:0 (FP16)...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
    model.eval()

    # Prompt variants for target concepts
    # Targets for person:
    target_person_prompts = [
        "a person",
        "a photo of a person",
        "a pedestrian walking",
        "a walking person"
    ]
    # Targets for bag:
    target_bag_prompts = [
        "a bag",
        "a tote bag",
        "a shoulder bag",
        "a backpack or bag"
    ]

    # Shared negative distractors
    distractors = [
        "a motor vehicle or car",
        "a dog or pet animal",
        "a tree or plant",
        "an outdoor gate or entrance"
    ]

    # Evaluate each prompt variant family against distractors
    prompt_sets = {
        "variant_1_bare": {
            "person": "a person",
            "tote bag": "a tote bag",
        },
        "variant_2_photo_of": {
            "person": "a photo of a person",
            "tote bag": "a photo of a tote bag",
        },
        "variant_3_descriptive": {
            "person": "a pedestrian walking",
            "tote bag": "a shoulder bag or tote bag",
        },
        "variant_4_composite": {
            "person": "a walking person",
            "tote bag": "a carrying bag",
        }
    }

    print("\n" + "=" * 70)
    print("EVALUATION OVER REAL CROPS (SigLIP Tokenization: max_length=64, padding='max_length')")
    print("=" * 70)

    # We will test each prompt variant family across all crops
    results_by_variant = {v: [] for v in prompt_sets}

    for variant_name, targets in prompt_sets.items():
        print(f"\n--- Testing Prompt Variant: {variant_name} ---")
        correct_top1_count = 0
        total_eval_count = 0

        for crop_info in crops[:15]: # evaluate at least 10-15 crops
            cid = crop_info["crop_id"]
            label = crop_info["label"]
            img = crop_info["image"]

            # Determine target prompt based on detected label
            target_p = targets.get(label, targets["person"])
            candidate_texts = [target_p] + distractors

            # Tokenize strictly with max_length=64, padding="max_length", truncation=True
            with torch.no_grad():
                text_inputs = processor(
                    text=candidate_texts,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda:0")
                text_out = model.get_text_features(**text_inputs)
                text_emb = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
                if isinstance(text_emb, tuple):
                    text_emb = text_emb[0]
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

                img_inputs = processor(images=img, return_tensors="pt").to("cuda:0")
                if "pixel_values" in img_inputs:
                    img_inputs["pixel_values"] = img_inputs["pixel_values"].to(torch.float16)
                img_out = model.get_image_features(**img_inputs)
                img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
                if isinstance(img_emb, tuple):
                    img_emb = img_emb[0]
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                sims = (img_emb @ text_emb.T)[0].tolist()
                ranked = sorted(zip(candidate_texts, sims), key=lambda x: x[1], reverse=True)

                # Check if target prompt is Rank 1
                is_top1 = (ranked[0][0] == target_p)
                if is_top1:
                    correct_top1_count += 1
                total_eval_count += 1

                target_rank = [i for i, (q, _) in enumerate(ranked, 1) if q == target_p][0]

                print(f"Crop {cid:02d} [{label:8s} at {crop_info['timestamp_s']:.1f}s]: Target Rank={target_rank}/{len(candidate_texts)} (Score={sims[0]:+.4f}) | Best: \"{ranked[0][0]}\" ({ranked[0][1]:+.4f})")

        acc = (correct_top1_count / total_eval_count) * 100 if total_eval_count > 0 else 0
        results_by_variant[variant_name] = {
            "correct": correct_top1_count,
            "total": total_eval_count,
            "accuracy": acc
        }
        print(f"--> Variant '{variant_name}' Top-1 Accuracy: {correct_top1_count}/{total_eval_count} ({acc:.1f}%)")

    print("\n" + "=" * 70)
    print("PROMPT VARIANT COMPARISON SUMMARY:")
    print("=" * 70)
    for vname, res in results_by_variant.items():
        print(f"  {vname:25s}: {res['correct']}/{res['total']} top-1 ({res['accuracy']:.1f}%)")

if __name__ == "__main__":
    main()
