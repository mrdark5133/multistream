import argparse
import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Set, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingest.pipeline import IngestPipeline
from src.index.db import get_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WATCH] %(message)s"
)
logger = logging.getLogger("multistream.watch")


def is_file_stable(file_path: Path, wait_s: float = 1.0) -> bool:
    """Check if a file size is stable (i.e. not actively being written by ffmpeg or recorder)."""
    try:
        size1 = file_path.stat().st_size
        time.sleep(wait_s)
        size2 = file_path.stat().st_size
        return size1 == size2 and size1 > 0
    except Exception:
        return False


def scan_and_ingest(
    watch_dirs: List[Path],
    pipeline: IngestPipeline,
    processed_set: Set[str],
    manifest_path: Path | None = None
) -> int:
    """Scan watch directories for unindexed video files and ingest them."""
    ingested_count = 0
    extensions = ("*.mp4", "*.mkv", "*.mov", "*.avi")

    for w_dir in watch_dirs:
        if not w_dir.exists():
            continue

        for ext in extensions:
            for video_file in sorted(w_dir.glob(ext)):
                # Ignore temp / invalid files
                if "_invalid" in video_file.name or "orientation_check" in str(video_file) or video_file.name.startswith("."):
                    continue

                abs_path_str = str(video_file.resolve())
                rel_path_str = str(video_file.relative_to(ROOT_DIR)) if video_file.is_relative_to(ROOT_DIR) else str(video_file)

                # Skip if already processed in this process session
                if abs_path_str in processed_set:
                    continue

                # Check SQLite database directly to prevent duplicate indexing
                v_record = get_video(pipeline.conn, rel_path_str) or get_video(pipeline.conn, abs_path_str)
                if v_record:
                    processed_set.add(abs_path_str)
                    continue

                # Ensure file is done writing
                logger.info(f"Detected new video file: {video_file.name}. Verifying write stability...")
                if not is_file_stable(video_file, wait_s=0.5):
                    logger.info(f"File {video_file.name} is still actively writing. Skipping for next poll cycle.")
                    continue

                logger.info(f"Starting incremental ingest for: {video_file.name}")
                try:
                    stats = pipeline.ingest_video(
                        video_path=video_file,
                        manifest_path=manifest_path,
                        force=False
                    )
                    processed_set.add(abs_path_str)
                    ingested_count += 1
                    logger.info(
                        f"Successfully indexed {video_file.name} | Camera: {stats.get('camera')} | "
                        f"Tracks: {stats.get('num_tracks')} | Frames: {stats.get('num_frames')} | Wall: {stats.get('wall_time_s', 0):.2f}s"
                    )
                except Exception as e:
                    logger.error(f"Error ingesting {video_file.name}: {e}")

    return ingested_count


def main():
    parser = argparse.ArgumentParser(description="MULTIStream Unified Folder Watcher")
    parser.add_argument("--watch-dirs", nargs="+", default=["footage"], help="One or more directories to watch")
    parser.add_argument("--index-dir", type=str, default="index_base", help="Index storage directory")
    parser.add_argument("--manifest", type=str, default="footage/manifest.json", help="Path to camera manifest")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run single scan pass and exit (batch/test mode)")
    args = parser.parse_args()

    watch_dirs = [Path(d).resolve() for d in args.watch_dirs]
    index_dir = Path(args.index_dir).resolve()
    db_path = index_dir / "index.db"
    snaps_dir = index_dir / "snapshots"
    manifest_path = Path(args.manifest).resolve() if args.manifest and Path(args.manifest).exists() else None

    logger.info("=" * 70)
    logger.info("MULTIStream Unified Folder Watcher")
    logger.info(f"Watching Directories: {[str(d) for d in watch_dirs]}")
    logger.info(f"Target Database:      {db_path}")
    logger.info(f"Poll Interval:        {args.poll_interval}s")
    logger.info("=" * 70)

    pipeline = IngestPipeline(db_path=db_path, snapshots_dir=snaps_dir)
    processed_set: Set[str] = set()

    if args.once:
        count = scan_and_ingest(watch_dirs, pipeline, processed_set, manifest_path=manifest_path)
        logger.info(f"Single pass complete. Ingested {count} new videos.")
        return

    logger.info("Starting watch loop. Press Ctrl+C to terminate.")
    try:
        while True:
            scan_and_ingest(watch_dirs, pipeline, processed_set, manifest_path=manifest_path)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Watch service terminated by user.")


if __name__ == "__main__":
    main()
