import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.pipeline import IngestPipeline


def main():
    parser = argparse.ArgumentParser(description="MULTIStream Video Ingestion CLI")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file to ingest")
    parser.add_argument("--footage-dir", type=str, default="footage", help="Directory of videos to ingest")
    parser.add_argument("--index-dir", type=str, default="index_base", help="Index storage directory")
    parser.add_argument("--embedder", type=str, default="google/siglip-base-patch16-224", help="SigLIP model ID")
    parser.add_argument("--stride", type=int, default=5, help="Detection frame sampling stride")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--frame-interval", type=float, default=2.0, help="Whole-frame sampling interval in seconds")
    parser.add_argument("--camera", type=str, default=None, help="Manual camera name override")
    parser.add_argument("--start-time", type=str, default=None, help="Manual start time ISO override")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.json")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if already indexed")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    db_path = index_dir / "index.db"
    snapshots_dir = index_dir / "snapshots"

    print("=" * 80)
    print(f"MULTIStream Ingestion Pipeline")
    print(f"Index DB:      {db_path}")
    print(f"Snapshots:     {snapshots_dir}")
    print(f"Embedder:      {args.embedder}")
    print(f"Stride:        {args.stride}")
    print(f"Imgsz:         {args.imgsz}")
    print("=" * 80)

    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        embedder_name=args.embedder
    )

    videos = []
    if args.video:
        videos.append(Path(args.video))
    else:
        footage_dir = Path(args.footage_dir)
        # Find all MP4 files, excluding invalid clips
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for vf in sorted(footage_dir.glob(ext)):
                if "_invalid" in str(vf) or "orientation_check" in str(vf) or "test_gate" in str(vf):
                    continue
                videos.append(vf)

    if not videos:
        print("No video files found to ingest.")
        return

    print(f"Found {len(videos)} video(s) for ingestion.")
    results = []

    for v in videos:
        print(f"\n--- Ingesting: {v} ---")
        res = pipeline.ingest_video(
            video_path=v,
            manifest_path=args.manifest,
            manual_camera=args.camera,
            manual_start=args.start_time,
            stride=args.stride,
            conf_thresh=args.conf,
            imgsz=args.imgsz,
            frame_sample_interval_s=args.frame_interval,
            force=args.force
        )
        results.append(res)
        
        if res.get("status") == "skipped":
            print(f"  [SKIPPED] Video already indexed at {res.get('indexed_at')}")
        else:
            print(f"  [INDEXED] Camera:        {res.get('camera')}")
            print(f"            Start Time:    {res.get('start_time')} (source: {res.get('start_source')})")
            print(f"            Duration:      {res.get('duration_s'):.2f}s | Rotation: {res.get('rotation')} deg")
            print(f"            Tracks Extr:   {res.get('num_tracks')}")
            print(f"            Frames Extr:   {res.get('num_frames')}")
            print(f"            Wall Time:     {res.get('wall_time_s'):.2f}s")
            print(f"            VRAM Used:     {res.get('vram_used_mib')} MiB")

    print("\n" + "=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)
    total_tracks = sum(r.get("num_tracks", 0) for r in results if r.get("status") == "indexed")
    total_frames = sum(r.get("num_frames", 0) for r in results if r.get("status") == "indexed")
    total_time = sum(r.get("wall_time_s", 0.0) for r in results if r.get("status") == "indexed")
    print(f"Total videos processed: {len(results)}")
    print(f"Total tracks indexed:   {total_tracks}")
    print(f"Total frames indexed:   {total_frames}")
    print(f"Total ingest wall time: {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
