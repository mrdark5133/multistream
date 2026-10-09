import time
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
live_vocab_path = ROOT_DIR / "config" / "live_vocab.yaml"

with open(live_vocab_path, "r", encoding="utf-8") as f:
    vocab_data = yaml.safe_load(f)

classes = list(vocab_data["classes"])
print(f"Initial live vocabulary ({len(classes)} classes):")
print(classes)

# Load model
t0 = time.perf_counter()
yolo = YOLO("yolov8s-worldv2.pt")
yolo.set_classes(classes)
if torch.cuda.is_available():
    yolo.to("cuda:0")
yolo.float()
# Ensure clip_model is ready on cuda
yolo.model.clip_model = None
load_time = time.perf_counter() - t0
print(f"Model load time: {load_time*1000:.2f} ms")

query_phrases = [
    "frooti juice box",
    "monster energy can",
    "power bank",
    "wireless mouse",
    "usb cable"
]

print("\n--- Vocabulary Extension Delay Benchmark ---")
for phrase in query_phrases:
    phrase_clean = phrase.strip().lower()
    if phrase_clean not in classes:
        t_start = time.perf_counter()
        extended_classes = classes + [phrase_clean]
        # Keep clip_model on cuda
        yolo.model.clip_model = None
        yolo.set_classes(extended_classes)
        t_delay = (time.perf_counter() - t_start) * 1000.0
        classes = extended_classes
        print(f"Added '{phrase_clean}': Delay = {t_delay:.2f} ms (Total classes now: {len(classes)})")
    else:
        print(f"'{phrase_clean}' already present in vocabulary.")
