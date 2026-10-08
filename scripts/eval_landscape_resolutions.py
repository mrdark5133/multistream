import cv2
import yaml
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

vocab = yaml.safe_load(open('config/vocab.yaml', 'r', encoding='utf-8'))['classes']
yolo = YOLO('yolov8s-worldv2.pt')
yolo.set_classes(vocab)

out_dir = Path('footage/orientation_check')
out_dir.mkdir(parents=True, exist_ok=True)

clips = [
    ("test_landscape.mp4", [1.0, 7.5, 14.0], ["start", "mid", "end"]),
    ("test_landscape2.mp4", [1.0, 14.5, 28.0], ["start", "mid", "end"])
]

resolutions = [640, 1280]

results = {}

for clip_name, timestamps, labels in clips:
    clip_path = f"footage/{clip_name}"
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results[clip_name] = {}
    
    for t, pos in zip(timestamps, labels):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read {clip_name} at {t}s")
            continue
        
        results[clip_name][pos] = {"timestamp": t}
        
        for imgsz in resolutions:
            res = yolo.predict(frame, conf=0.25, imgsz=imgsz, device='cuda:0', verbose=False)[0]
            counts = Counter()
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                cname = vocab[cls_id] if cls_id < len(vocab) else f'class_{cls_id}'
                counts[cname] += 1
            
            # Save annotated frame
            annotated_img = res.plot()
            stem = Path(clip_name).stem
            save_path = out_dir / f"annotated_{imgsz}_{stem}_{pos}.jpg"
            cv2.imwrite(str(save_path), annotated_img)
            
            results[clip_name][pos][imgsz] = {
                "total": sum(counts.values()),
                "counts": dict(counts),
                "saved_path": str(save_path)
            }
    cap.release()

print("\n" + "="*80)
print("YOLO-WORLD DETECTION BENCHMARK: LANDSCAPE CLIPS (imgsz=640 vs imgsz=1280)")
print("="*80)

for clip_name, pos_data in results.items():
    print(f"\n### Clip: footage/{clip_name}")
    for pos, data in pos_data.items():
        t = data["timestamp"]
        print(f"\n--- Position: {pos} (t={t:.1f}s) ---")
        for imgsz in resolutions:
            info = data[imgsz]
            print(f"  imgsz={imgsz}: Total Detections = {info['total']}")
            for cname, cnt in sorted(info['counts'].items(), key=lambda x: -x[1]):
                print(f"    - {cname}: {cnt}")
            print(f"    [Annotated image saved: {info['saved_path']}]")

print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)
print(f"{'Clip':<20} | {'Frame':<8} | {'Timestamp':<10} | {'imgsz=640':<12} | {'imgsz=1280':<12} | {'Ratio (1280/640)':<16}")
print("-" * 88)
for clip_name, pos_data in results.items():
    for pos, data in pos_data.items():
        t = data["timestamp"]
        c640 = data[640]["total"]
        c1280 = data[1280]["total"]
        ratio = f"{c1280 / c640:.2f}x" if c640 > 0 else "N/A"
        print(f"{clip_name:<20} | {pos:<8} | {t:<10.1f} | {c640:<12} | {c1280:<12} | {ratio:<16}")
