import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
from ultralytics import YOLO

# 1. Grab 1 frame
cap = cv2.VideoCapture("footage/mobile_cam02_20261009_024019.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame_bgr = cap.read()
cap.release()
assert ret, "Failed to read frame"
H, W = frame_bgr.shape[:2]
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)

classes = [
    "a bluetooth speaker",
    "a phone charger",
    "a laptop",
    "a computer monitor",
    "a soda can",
    "a juice box",
    "a juice bottle"
]

print("Testing YOLO-World...")
yolo = YOLO("yolov8s-worldv2.pt")
yolo.set_classes(classes)
yolo.to("cuda:0")
res = yolo.predict(frame_bgr, conf=0.15, imgsz=640, verbose=False)
boxes = res[0].boxes
print(f"YOLO-World 640 detections: {len(boxes) if boxes is not None else 0}")
del yolo
torch.cuda.empty_cache()

print("Testing Grounding DINO...")
gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda:0")
gd_text = ". ".join(classes) + "."
gd_inputs = gd_proc(images=pil_img, text=gd_text, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    gd_outputs = gd_model(**gd_inputs)
gd_res = gd_proc.post_process_grounded_object_detection(
    gd_outputs,
    gd_inputs.input_ids,
    threshold=0.20,
    text_threshold=0.20,
    target_sizes=[(H, W)],
    text_labels=[classes]
)[0]
print(f"Grounding DINO detections: {len(gd_res['boxes'])}")
for score, label, box in zip(gd_res["scores"], gd_res["labels"], gd_res["boxes"]):
    print(f"  {label} ({score:.2f}) {box.int().tolist()}")
del gd_model, gd_proc
torch.cuda.empty_cache()

print("Testing OWLv2...")
owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda:0")
owl_inputs = owl_proc(text=[classes], images=pil_img, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    owl_outputs = owl_model(**owl_inputs)
owl_res = owl_proc.post_process_grounded_object_detection(
    outputs=owl_outputs,
    target_sizes=[(H, W)],
    threshold=0.15
)[0]
print(f"OWLv2 detections: {len(owl_res['boxes'])}")
for score, label_idx, box in zip(owl_res["scores"], owl_res["labels"], owl_res["boxes"]):
    lbl = classes[label_idx.item()] if label_idx.item() < len(classes) else "unknown"
    print(f"  {lbl} ({score:.2f}) {box.int().tolist()}")
del owl_model, owl_proc
torch.cuda.empty_cache()

print("All single-frame tests passed!")
