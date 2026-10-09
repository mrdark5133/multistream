import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from ultralytics import YOLO

from src.utils.vram import check_vram_headroom


# Mapping of COCO class IDs to keep for CCTV, traffic, and indoor environments
COCO_KEEP_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    56: "chair",
    57: "couch",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    74: "clock"
}

COCO_NAMES_SET = set(COCO_KEEP_CLASSES.values())


def get_object_group(label: str) -> str:
    """Classify label into coarse group: person | vehicle | object."""
    lbl = label.lower().strip()
    if lbl.startswith("a "):
        lbl = lbl[2:].strip()
    elif lbl.startswith("an "):
        lbl = lbl[3:].strip()
    if lbl in {
        "person", "pedestrian", "cyclist", "child", "security guard",
        "delivery worker", "construction worker"
    }:
        return "person"
    if lbl in {
        "car", "sedan", "suv", "van", "pickup truck", "truck", "semi-truck",
        "bus", "motorcycle", "motorbike", "bicycle", "electric scooter",
        "autorickshaw", "ambulance", "police car", "fire truck", "forklift"
    }:
        return "vehicle"
    return "object"



def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class HybridDetector:
    """
    Hybrid Object Detector:
    1. Primary: COCO-trained YOLO11 (yolo11s.pt) restricted to KEEP classes for maximum precision.
    2. Secondary: YOLO-World (yolov8s-worldv2.pt) configured only for non-COCO domain vocabulary.
    """
    def __init__(
        self,
        coco_weights: str = "yolo11s.pt",
        world_weights: str = "yolov8s-worldv2.pt",
        vocab_path: str = "config/vocab.yaml",
        device: str = "cuda:0",
        conf_thresh: float = 0.25,
        iou_suppress_thresh: float = 0.40
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_suppress_thresh = iou_suppress_thresh
        
        # 1. Load Primary COCO YOLO11
        self.yolo_coco = YOLO(coco_weights)
        if device.startswith("cuda") and torch.cuda.is_available():
            self.yolo_coco.to(device)
        self.coco_keep_ids = list(COCO_KEEP_CLASSES.keys())
        
        # 2. Extract non-COCO vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = yaml.safe_load(f)
        full_vocab = vocab_data.get("classes", [])
        self.non_coco_classes = [c for c in full_vocab if c.lower() not in COCO_NAMES_SET]
        
        # 3. Load Secondary YOLO-World on CPU first, set non-COCO classes, then move to GPU
        self.yolo_world = YOLO(world_weights)
        self.yolo_world.set_classes(self.non_coco_classes)
        if device.startswith("cuda") and torch.cuda.is_available():
            self.yolo_world.to(device)
        self.yolo_world.float()

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 640) -> List[Dict[str, Any]]:
        """
        Run both detectors and return non-overlapping fused detections.
        Each detection: {"bbox": [x1, y1, x2, y2], "conf": float, "label": str, "group": str, "source": str}
        """
        detections: List[Dict[str, Any]] = []
        coco_boxes_np = []

        # 1. Primary COCO detections
        res_coco = self.yolo_coco.predict(
            frame_bgr,
            classes=self.coco_keep_ids,
            conf=self.conf_thresh,
            imgsz=imgsz,
            verbose=False
        )[0]

        if res_coco.boxes is not None and len(res_coco.boxes) > 0:
            for b in res_coco.boxes:
                cls_id = int(b.cls[0].item())
                label = COCO_KEEP_CLASSES.get(cls_id, f"coco_{cls_id}")
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), int(xyxy[2]), int(xyxy[3])
                
                if (x2 - x1) * (y2 - y1) <= 0:
                    continue

                box = [x1, y1, x2, y2]
                detections.append({
                    "bbox": box,
                    "conf": conf,
                    "label": label,
                    "group": get_object_group(label),
                    "source": "coco_yolo11"
                })
                coco_boxes_np.append(np.array(box))

        # 2. Secondary non-COCO YOLO-World detections
        res_world = self.yolo_world.predict(
            frame_bgr,
            conf=self.conf_thresh,
            imgsz=imgsz,
            verbose=False
        )[0]

        if res_world.boxes is not None and len(res_world.boxes) > 0:
            for b in res_world.boxes:
                cls_id = int(b.cls[0].item())
                label = self.non_coco_classes[cls_id] if cls_id < len(self.non_coco_classes) else f"world_{cls_id}"
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), int(xyxy[2]), int(xyxy[3])
                
                if (x2 - x1) * (y2 - y1) <= 0:
                    continue

                w_box = np.array([x1, y1, x2, y2])
                
                # Check overlap with existing COCO detections
                is_overlap = False
                for c_box in coco_boxes_np:
                    if box_iou(w_box, c_box) > self.iou_suppress_thresh:
                        is_overlap = True
                        break

                if not is_overlap:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "label": label,
                        "group": get_object_group(label),
                        "source": "yolo_world"
                    })

        # 3. Class-group NMS (IoU 0.60) across fused detections
        return self._apply_class_group_nms(detections, iou_thresh=0.60)

    @staticmethod
    def _apply_class_group_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.60) -> List[Dict[str, Any]]:
        """Group detections by coarse category group and suppress boxes with IoU >= iou_thresh."""
        if len(detections) <= 1:
            return detections

        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for d in detections:
            grp = d.get("group", get_object_group(d["label"]))
            by_group.setdefault(grp, []).append(d)

        kept: List[Dict[str, Any]] = []
        for grp, group_dets in by_group.items():
            group_dets.sort(key=lambda x: x["conf"], reverse=True)
            group_kept: List[Dict[str, Any]] = []
            for d in group_dets:
                b = np.array(d["bbox"])
                suppressed = False
                for kd in group_kept:
                    kb = np.array(kd["bbox"])
                    if box_iou(b, kb) >= iou_thresh:
                        suppressed = True
                        break
                if not suppressed:
                    group_kept.append(d)
            kept.extend(group_kept)
        return kept


class GroundingDinoDetector:
    """
    Open-vocabulary zero-shot detector using Grounding-DINO-Tiny.
    Designed for granular desk/indoor objects and user-specified vocabularies.
    """
    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        vocab_path: str = "config/test_vocab.yaml",
        device: str = "cuda:0",
        conf_thresh: float = 0.20,
        text_thresh: float = 0.20,
        classes: Optional[List[str]] = None
    ):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device
        self.conf_thresh = conf_thresh
        self.text_thresh = text_thresh

        if classes is not None:
            raw_classes = classes
        else:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = yaml.safe_load(f)
            raw_classes = vocab_data.get("classes", [])

        self.raw_classes = []
        for c in raw_classes:
            c_str = c.strip()
            for pfx in ("a ", "an ", "the "):
                if c_str.lower().startswith(pfx):
                    c_str = c_str[len(pfx):].strip()
            if c_str and c_str not in self.raw_classes:
                self.raw_classes.append(c_str)

        self.prompt_text = " . ".join(self.raw_classes) + " ."

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()

    def _resolve_clean_label(self, raw_pred: str) -> str:
        s = raw_pred.strip().lower()
        for pfx in ("a ", "an ", "the "):
            if s.startswith(pfx):
                s = s[len(pfx):].strip()

        # 1. Exact match against canonical classes
        for c in self.raw_classes:
            if s == c.lower():
                return c

        # 2. Strict semantic keyword mapping for desk objects
        if "juice" in s:
            return "juice bottle" if "bottle" in s else "juice box"
        if "extension" in s or "power strip" in s:
            return "extension board"
        if "power bank" in s:
            return "power bank"
        if "charger" in s or "adapter" in s:
            return "phone charger"
        if "speaker" in s:
            return "bluetooth speaker"
        if "monitor" in s or "display" in s:
            return "computer monitor"
        if "laptop" in s or "notebook" in s:
            return "laptop"
        if "soda" in s:
            return "soda can"
        if "water" in s or ("bottle" in s and "juice" not in s):
            return "water bottle"

        # 3. For any other generic prediction (e.g. 'box', 'board', 'strip'),
        # return as is or check if it exactly equals a canonical class.
        for c in self.raw_classes:
            if c.lower() in s:
                return c

        return s

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 640) -> List[Dict[str, Any]]:
        from PIL import Image

        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = self.processor(images=pil_img, text=self.prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.conf_thresh,
            text_threshold=self.text_thresh,
            target_sizes=[(H, W)],
            text_labels=[self.raw_classes]
        )[0]

        detections: List[Dict[str, Any]] = []
        for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"]):
            matched_lbl = self._resolve_clean_label(lb)

            b_np = bx.int().cpu().numpy().tolist()
            x1 = max(0, min(W, b_np[0]))
            y1 = max(0, min(H, b_np[1]))
            x2 = max(0, min(W, b_np[2]))
            y2 = max(0, min(H, b_np[3]))

            if (x2 - x1) * (y2 - y1) <= 0:
                continue

            conf = float(sc.item())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "label": matched_lbl,
                "group": get_object_group(matched_lbl),
                "source": "grounding_dino"
            })

        return HybridDetector._apply_class_group_nms(detections, iou_thresh=0.50)

