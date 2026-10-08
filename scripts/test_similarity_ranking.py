"""
Phase 0 Task 7: Semantic Similarity Ranking Validation
Embed real images/crops and demonstrate sensible image-text similarity ranking.
"""
import os
import sys
import yaml
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLOWorld
from transformers import AutoProcessor, AutoModel

def main():
    print("=" * 70)
    print("Phase 0 Task 7: Semantic Similarity Ranking Validation")
    print("=" * 70)

    image_path = "footage/sample_frame.jpg"
    if not Path(image_path).exists():
        print(f"Error: {image_path} does not exist!")
        sys.exit(1)

    # 1. Run YOLO to find real objects in the frame
    print(f"Loading YOLO-World to detect objects in {image_path}...")
    yolo = YOLOWorld("yolov8s-worldv2.pt")
    vocab_path = Path("config/vocab.yaml")
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            classes = yaml.safe_load(f).get("classes", [])
        yolo.set_classes(classes)

    results = yolo.predict(image_path, device="cuda:0", verbose=False)
    boxes = results[0].boxes
    print(f"Detected {len(boxes)} objects:")
    
    crops = []
    crop_labels = []
    orig_img = Image.open(image_path).convert("RGB")
    crops_dir = Path("footage/crops")
    crops_dir.mkdir(exist_ok=True)

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        label = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
        conf = float(box.conf[0].item())
        xyxy = [int(x) for x in box.xyxy[0].tolist()]
        print(f"  [{i}] {label} (conf: {conf:.3f}, bbox: {xyxy})")
        crop = orig_img.crop(xyxy)
        crop_save_path = crops_dir / f"crop_{i}_{label}.jpg"
        crop.save(crop_save_path)
        crops.append((f"crop_{i}_{label}", crop, label))

    # Also include the whole frame
    crops.append(("whole_frame", orig_img, "scene"))

    # 2. Evaluate with SigLIP
    for model_id in ["google/siglip-base-patch16-224", "google/siglip-so400m-patch14-384"]:
        print("\n" + "-" * 70)
        print(f"Evaluating Embedder: {model_id}")
        print("-" * 70)
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
        model.eval()

        test_queries = [
            "a person",
            "a pedestrian walking",
            "a motor vehicle or car",
            "a dog or pet animal",
            "a tree or plant",
            "an outdoor gate or entrance"
        ]

        print(f"\nComputing normalized embeddings and cosine similarity against: {test_queries}")
        
        with torch.no_grad():
            text_inputs = processor(text=test_queries, padding="max_length", return_tensors="pt").to("cuda:0")
            text_out = model.get_text_features(**text_inputs)
            text_emb = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
            if isinstance(text_emb, tuple):
                text_emb = text_emb[0]
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            for name, img, ground_label in crops:
                img_inputs = processor(images=img, return_tensors="pt").to("cuda:0")
                if "pixel_values" in img_inputs:
                    img_inputs["pixel_values"] = img_inputs["pixel_values"].to(torch.float16)
                img_out = model.get_image_features(**img_inputs)
                img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
                if isinstance(img_emb, tuple):
                    img_emb = img_emb[0]
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                # Cosine similarities
                sims = (img_emb @ text_emb.T)[0].tolist()
                ranked = sorted(zip(test_queries, sims), key=lambda x: x[1], reverse=True)
                
                print(f"\nImage: '{name}' (Expected category: '{ground_label}'):")
                for rank, (q, score) in enumerate(ranked, 1):
                    print(f"   Rank {rank}: {score:+.4f} | \"{q}\"")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Task 7 Ranking Validation Finished Successfully")
    print("=" * 70)

if __name__ == "__main__":
    main()
