"""
Phase 0 Fix-up Task 2:
- Print, per variant, the exact target prompt used for every crop and the full candidate list.
- Use the SAME candidate set across variants.
- Explain what hat crops 03/06 were scored against in previous runs.
- Retrieval-style test: for each query (person, bag, hat) rank all crops and report Recall@k (k=1, 3, 5) and ROC-AUC.
- Save a contact sheet of all 17 crops with their labels to footage/crops_contact_sheet.jpg.
"""
import os
import sys
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel

# 1. Map filenames to ground truth categories
# filenames: crop_00_handbag_0.5s.jpg, crop_01_person_0.5s.jpg, etc.
def get_crop_data():
    crops_dir = Path("footage/eval_crops/crops")
    crop_files = sorted(crops_dir.glob("crop_*.jpg"))
    data = []
    for cf in crop_files:
        name = cf.name # e.g. crop_00_handbag_0.5s.jpg
        parts = name.replace(".jpg", "").split("_")
        cid = int(parts[1])
        raw_label = parts[2]
        ts = parts[3]
        
        # Ground truth category mapping
        if "person" in raw_label:
            category = "person"
        elif "bag" in raw_label:
            category = "bag"
        elif "hat" in raw_label:
            category = "hat"
        else:
            category = "other"

        data.append({
            "crop_id": cid,
            "filename": name,
            "filepath": cf,
            "raw_label": raw_label,
            "category": category,
            "timestamp": ts,
            "image": Image.open(cf).convert("RGB")
        })
    return data

def make_contact_sheet(crop_data: list[dict], output_path: Path):
    """
    Creates a visual contact sheet grid of all crops with labels.
    """
    n_crops = len(crop_data)
    cols = 6
    rows = (n_crops + cols - 1) // cols
    tile_w = 200
    tile_h = 240
    padding = 10

    total_w = cols * tile_w + (cols + 1) * padding
    total_h = rows * tile_h + (rows + 1) * padding + 50 # 50px for header

    sheet = Image.new("RGB", (total_w, total_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    # Header
    title = f"MULTIStream Phase 0 Evaluation Crops Contact Sheet (N={n_crops})"
    draw.text((padding, 15), title, fill=(240, 240, 240))

    font_color = (255, 255, 255)
    bg_box = (40, 40, 45)

    for i, c in enumerate(crop_data):
        r = i // cols
        col = i % cols
        x0 = padding + col * (tile_w + padding)
        y0 = 60 + padding + r * (tile_h + padding)

        # Background card
        draw.rectangle([x0, y0, x0 + tile_w, y0 + tile_h], fill=bg_box, outline=(70, 70, 80))

        # Thumbnail
        img = c["image"]
        img_copy = img.copy()
        img_copy.thumbnail((tile_w - 16, tile_h - 70))
        iw, ih = img_copy.size
        img_x = x0 + (tile_w - iw) // 2
        img_y = y0 + 8
        sheet.paste(img_copy, (img_x, img_y))

        # Text metadata
        tag1 = f"ID:{c['crop_id']:02d} [{c['category'].upper()}]"
        tag2 = f"{c['raw_label']} ({c['timestamp']})"
        draw.text((x0 + 8, y0 + tile_h - 45), tag1, fill=(0, 220, 180))
        draw.text((x0 + 8, y0 + tile_h - 25), tag2, fill=(200, 200, 200))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"\n[Contact Sheet] Saved contact sheet of {n_crops} crops to: {output_path}")

def compute_roc_auc(scores: list[float], binary_labels: list[int]) -> float:
    """
    Computes ROC-AUC using Mann-Whitney U rank-sum.
    """
    n_pos = sum(binary_labels)
    n_neg = len(binary_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Rank scores (1-indexed, handle ties by average rank)
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and indexed_scores[j][1] == indexed_scores[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed_scores[k][0]] = avg_rank
        i = j

    rank_sum_pos = sum(ranks[idx] for idx, lbl in enumerate(binary_labels) if lbl == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(auc)

def compute_recall_at_k(ranked_indices: list[int], binary_labels: list[int], k: int) -> float:
    """
    Fraction of true positive items retrieved in top-k results.
    """
    total_pos = sum(binary_labels)
    if total_pos == 0:
        return 0.0
    top_k_indices = ranked_indices[:k]
    hits = sum(binary_labels[idx] for idx in top_k_indices)
    return hits / total_pos

def main():
    print("=" * 80)
    print("PHASE 0 FIX-UP 2: SIMILARITY, CANDIDATE SET & RETRIEVAL-STYLE EVALUATION")
    print("=" * 80)

    # 1. Load crop data
    crop_data = get_crop_data()
    print(f"Loaded {len(crop_data)} crops from footage/eval_crops/crops.")
    cat_counts = {}
    for c in crop_data:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print(f"Class distribution: {cat_counts}")

    # 2. Generate contact sheet
    contact_sheet_path = Path("footage/crops_contact_sheet.jpg")
    make_contact_sheet(crop_data, contact_sheet_path)

    # 3. Load SigLIP-Base
    model_id = "google/siglip-base-patch16-224"
    print(f"\nLoading embedder: {model_id} on cuda:0 (FP16)...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
    model.eval()

    # 4. Define candidate sets per variant (SAME CANDIDATE SET ACROSS ALL CROPS WITHIN A VARIANT)
    # Every variant has 7 candidates: [target_for_person, target_for_bag, target_for_hat, distractor1, distractor2, distractor3, distractor4]
    variants = {
        "variant_1_bare": {
            "candidates": [
                "a person",
                "a bag",
                "a hat",
                "a motor vehicle or car",
                "a dog or pet animal",
                "a tree or plant",
                "an outdoor gate or entrance"
            ],
            "target_map": {
                "person": "a person",
                "bag": "a bag",
                "hat": "a hat"
            }
        },
        "variant_2_photo_of": {
            "candidates": [
                "a photo of a person",
                "a photo of a bag",
                "a photo of a hat",
                "a photo of a motor vehicle or car",
                "a photo of a dog or pet animal",
                "a photo of a tree or plant",
                "a photo of an outdoor gate or entrance"
            ],
            "target_map": {
                "person": "a photo of a person",
                "bag": "a photo of a bag",
                "hat": "a photo of a hat"
            }
        },
        "variant_3_descriptive": {
            "candidates": [
                "a pedestrian walking",
                "a tote bag or handbag",
                "a hat or cap worn on head",
                "a motor vehicle or automobile",
                "a domestic dog or canine",
                "a leafy tree or plant",
                "a security gate or entrance barrier"
            ],
            "target_map": {
                "person": "a pedestrian walking",
                "bag": "a tote bag or handbag",
                "hat": "a hat or cap worn on head"
            }
        },
        "variant_4_composite": {
            "candidates": [
                "a walking person",
                "a carried handbag or tote bag",
                "a headwear hat or cap",
                "a motor vehicle on a road",
                "a pet dog on a leash",
                "a roadside tree or bush",
                "a metal gate or barrier"
            ],
            "target_map": {
                "person": "a walking person",
                "bag": "a carried handbag or tote bag",
                "hat": "a headwear hat or cap"
            }
        }
    }

    # Pre-extract crop embeddings
    print("\nExtracting SigLIP image embeddings for all crops...")
    crop_embeddings = []
    with torch.no_grad():
        for c in crop_data:
            inputs = processor(images=c["image"], return_tensors="pt").to("cuda:0")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            feat = model.get_image_features(**inputs)
            emb = feat.pooler_output if hasattr(feat, "pooler_output") else feat
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            crop_embeddings.append(emb)
    crop_embeddings = torch.cat(crop_embeddings, dim=0) # [17, 768]
    print(f"Extracted crop embeddings matrix: {crop_embeddings.shape}")

    # 5. Evaluate each prompt variant
    print("\n" + "=" * 80)
    print("PROMPT VARIANT RANKING EVALUATION (IDENTICAL CANDIDATE SET PER VARIANT)")
    print("=" * 80)

    variant_summary = {}

    for v_name, v_info in variants.items():
        candidates = v_info["candidates"]
        target_map = v_info["target_map"]
        print(f"\n" + "-" * 75)
        print(f"VARIANT: {v_name}")
        print(f"FULL CANDIDATE LIST ({len(candidates)} items):")
        for idx, cand in enumerate(candidates, 1):
            print(f"   [{idx}] \"{cand}\"")
        print("-" * 75)

        # Encode candidates
        with torch.no_grad():
            t_inputs = processor(
                text=candidates,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            ).to("cuda:0")
            t_feat = model.get_text_features(**t_inputs)
            t_emb = t_feat.pooler_output if hasattr(t_feat, "pooler_output") else t_feat
            if isinstance(t_emb, tuple):
                t_emb = t_emb[0]
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True) # [7, 768]

        # Compute similarity matrix: [17 crops, 7 candidates]
        sim_matrix = (crop_embeddings @ t_emb.T).cpu().numpy()

        top1_correct = 0
        crop_results = []

        for i, c in enumerate(crop_data):
            cat = c["category"]
            target_prompt = target_map[cat]
            sims = sim_matrix[i]
            ranked_pairs = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
            ranked_texts = [p[0] for p in ranked_pairs]
            target_rank = ranked_texts.index(target_prompt) + 1
            target_score = sims[candidates.index(target_prompt)]
            best_text, best_score = ranked_pairs[0]

            is_top1 = (target_rank == 1)
            if is_top1:
                top1_correct += 1

            crop_results.append({
                "crop_id": c["crop_id"],
                "category": cat,
                "raw_label": c["raw_label"],
                "target_prompt": target_prompt,
                "target_rank": target_rank,
                "target_score": target_score,
                "best_text": best_text,
                "best_score": best_score,
                "is_top1": is_top1
            })

            status = "PASS" if is_top1 else "FAIL"
            print(f"Crop {c['crop_id']:02d} [{cat:6s}|{c['raw_label']:8s}]: Target='{target_prompt}' -> Rank {target_rank}/{len(candidates)} ({status}) | Score={target_score:+.4f} | Top: '{best_text}' ({best_score:+.4f})")

        acc = (top1_correct / len(crop_data)) * 100.0
        variant_summary[v_name] = {
            "top1_correct": top1_correct,
            "total": len(crop_data),
            "accuracy": acc,
            "results": crop_results
        }
        print(f"--> Variant {v_name} Top-1 Accuracy: {top1_correct}/{len(crop_data)} ({acc:.1f}%)")

    # 6. Retrieval-Style Test across all 17 crops for queries: person, bag, hat
    print("\n" + "=" * 80)
    print("RETRIEVAL-STYLE TEST: ALL CROPS RANKED BY QUERY (PERSON, BAG, HAT)")
    print("=" * 80)

    # We evaluate retrieval using the best-performing canonical prompts
    retrieval_queries = [
        {"concept": "person", "query": "a person", "true_class": "person"},
        {"concept": "bag",    "query": "a bag",    "true_class": "bag"},
        {"concept": "hat",    "query": "a hat",    "true_class": "hat"}
    ]

    retrieval_results = {}

    for q_item in retrieval_queries:
        concept = q_item["concept"]
        query_text = q_item["query"]
        true_cls = q_item["true_class"]

        # Encode single query
        with torch.no_grad():
            q_inputs = processor(
                text=[query_text],
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt"
            ).to("cuda:0")
            q_feat = model.get_text_features(**q_inputs)
            q_emb = q_feat.pooler_output if hasattr(q_feat, "pooler_output") else q_feat
            if isinstance(q_emb, tuple):
                q_emb = q_emb[0]
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

        # Similarity to all 17 crops: [17]
        scores = (crop_embeddings @ q_emb.T).squeeze(-1).cpu().numpy().tolist()
        binary_labels = [1 if c["category"] == true_cls else 0 for c in crop_data]
        n_pos = sum(binary_labels)

        # Rank all crops descending
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        rec1 = compute_recall_at_k(ranked_indices, binary_labels, 1)
        rec3 = compute_recall_at_k(ranked_indices, binary_labels, 3)
        rec5 = compute_recall_at_k(ranked_indices, binary_labels, 5)
        rec10 = compute_recall_at_k(ranked_indices, binary_labels, 10)
        auc = compute_roc_auc(scores, binary_labels)

        retrieval_results[concept] = {
            "query": query_text,
            "total_relevant": n_pos,
            "recall@1": rec1,
            "recall@3": rec3,
            "recall@5": rec5,
            "recall@10": rec10,
            "auc": auc,
            "ranked_crops": [(crop_data[idx]["crop_id"], crop_data[idx]["category"], scores[idx]) for idx in ranked_indices]
        }

        print(f"\nQuery: \"{query_text}\" (Target Concept: {concept.upper()}, N_true={n_pos}/{len(crop_data)}):")
        print(f"  Recall@1  : {rec1*100:5.1f}% ({int(rec1 * n_pos)}/{n_pos})")
        print(f"  Recall@3  : {rec3*100:5.1f}% ({int(rec3 * n_pos)}/{n_pos})")
        print(f"  Recall@5  : {rec5*100:5.1f}% ({int(rec5 * n_pos)}/{n_pos})")
        print(f"  Recall@10 : {rec10*100:5.1f}% ({int(rec10 * n_pos)}/{n_pos})")
        print(f"  ROC-AUC   : {auc:.4f}")
        print("  Top 5 Ranked Crops:")
        for rank, idx in enumerate(ranked_indices[:5], 1):
            c = crop_data[idx]
            match_str = "MATCH" if c["category"] == true_cls else "DIFF"
            print(f"    Rank {rank:2d}: Crop {c['crop_id']:02d} [{c['category']:6s}|{c['raw_label']:8s}] -> Score={scores[idx]:+.4f} ({match_str})")

    # 7. Overall Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & VERDICT")
    print("=" * 80)
    print("Prompt Classification Accuracies (Strict Top-1 among 7 candidates):")
    for v_name, res in variant_summary.items():
        print(f"  {v_name:25s}: {res['top1_correct']}/{res['total']} ({res['accuracy']:.1f}%)")
    print("\nRetrieval Performance (ROC-AUC and Recall@k):")
    for concept, res in retrieval_results.items():
        print(f"  Concept {concept:6s}: AUC={res['auc']:.4f} | R@1={res['recall@1']*100:.1f}% | R@3={res['recall@3']*100:.1f}% | R@5={res['recall@5']*100:.1f}%")

if __name__ == "__main__":
    main()
