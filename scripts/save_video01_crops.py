import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector
from src.ingest.detect_track import process_video_tracks_phase1b
from src.ingest.embed import SigLIPEmbedder

def main():
    clip = "footage/test_video01.mp4"
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()

    detector = HybridDetector(device="cuda:0")
    embedder = SigLIPEmbedder(device="cuda:0")

    tracks, _, dropped, merges = process_video_tracks_phase1b(
        video_path=clip,
        detector=detector,
        start_time_iso="2026-10-08T18:00:00",
        fps=fps,
        duration_s=dur,
        stride=2,
        tracker_name="bytetrack_tuned",
        embedder=embedder
    )

    out_dir = Path("eval/video01_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = [25, 26, 28, 35]
    found_crops = {}

    for t in tracks:
        if t.track_id in target_ids:
            fname = f"track_{t.track_id}_best.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), t.best_crop)
            found_crops[t.track_id] = {
                "path": str(fpath),
                "label": t.label,
                "hits": t.hits,
                "offset_best": t.offset_best,
                "quality": t.quality
            }

    print("Saved Crops for Tracks 25, 26, 28, 35:")
    for tid in target_ids:
        if tid in found_crops:
            info = found_crops[tid]
            print(f"  Track {tid:2d}: {info['path']} | Label: {info['label']} | Hits: {info['hits']} | Best Offset: {info['offset_best']:.3f}s | Quality: {info['quality']:.4f}")
        else:
            print(f"  Track {tid:2d}: NOT FOUND")

if __name__ == "__main__":
    main()
