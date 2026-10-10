import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch

from src.ingest.detector import GroundingDinoDetector
from src.ingest.pipeline import IngestPipeline

def main():
    root = Path(__file__).resolve().parent.parent
    db_path = root / "index_mobile" / "index.db"
    snapshots_dir = root / "index_mobile" / "snapshots"
    vocab_path = root / "config" / "test_vocab.yaml"

    videos_to_ingest = [
        ("footage/recorded/mobile_cam01_20261009_094250.mp4", "mobile_cam01"),
        ("footage/recorded/mobile_cam02_20261009_094356.mp4", "mobile_cam02"),
        ("footage/recorded/mobile_cam03_20261009_094427.mp4", "mobile_cam03"),
    ]

    print(f"Loading GroundingDINO with vocab: {vocab_path}...")
    detector = GroundingDinoDetector(
        vocab_path=str(vocab_path),
        device="cuda:0",
        conf_thresh=0.20,
        text_thresh=0.20
    )

    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        device="cuda:0",
        custom_detector=detector
    )

    total_new_tracks = 0
    for rel_path, camera in videos_to_ingest:
        full_path = root / rel_path
        if not full_path.exists():
            print(f"Skipping {rel_path} (not found)")
            continue
        print(f"\n--- Ingesting {rel_path} for camera {camera} ---")
        res = pipeline.ingest_video(
            video_path=full_path,
            manual_camera=camera,
            stride=5,
            conf_thresh=0.20,
            min_duration_s=0.3,
            min_hits=2,
            min_mean_conf=0.25,
            force=True
        )
        tracks_indexed = res.get("tracks_indexed", 0)
        total_new_tracks += tracks_indexed
        print(f"Done {rel_path}: {tracks_indexed} tracks indexed!")

    print(f"\nIngestion complete! Total new tracks indexed: {total_new_tracks}")

    del detector
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
