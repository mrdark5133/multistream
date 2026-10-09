import time
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.pipeline import IngestPipeline

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

def main():
    fresh_dir = Path("index_fresh")
    if fresh_dir.exists():
        shutil.rmtree(fresh_dir)
    fresh_dir.mkdir(parents=True, exist_ok=True)

    db_path = fresh_dir / "index.db"
    snapshots_dir = fresh_dir / "snapshots"

    print("=" * 80)
    print("COLD START INGESTION OF 5 DEV CLIPS INTO index_fresh/")
    print(f"Target DB:        {db_path}")
    print(f"Target Snapshots: {snapshots_dir}")
    print("=" * 80)

    # Initialize fresh pipeline (loads models into GPU: cold start)
    t_init0 = time.perf_counter()
    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        embedder_name="google/siglip-base-patch16-224",
        use_hybrid_detector=True
    )
    t_init = time.perf_counter() - t_init0
    print(f"Pipeline model load time: {t_init:.2f}s\n")

    results = []
    total_video_dur = 0.0
    total_wall_time = 0.0

    for clip_str in DEV_CLIPS:
        v_path = Path(clip_str)
        print(f"Ingesting: {v_path} ...")
        t0 = time.perf_counter()
        res = pipeline.ingest_video(
            video_path=v_path,
            stride=2,
            tracker_name="bytetrack_tuned",
            force=True
        )
        elapsed = time.perf_counter() - t0
        dur = res["duration_s"]
        ratio = elapsed / dur if dur > 0 else 0.0

        total_video_dur += dur
        total_wall_time += elapsed

        results.append({
            "clip": clip_str,
            "duration_s": dur,
            "wall_time_s": elapsed,
            "ratio": ratio,
            "tracks": res["num_tracks"],
            "frames": res["num_frames"]
        })
        print(f"  Done in {elapsed:.2f}s | Duration: {dur:.2f}s | Ratio (wall/dur): {ratio:.2f}x | Tracks: {res['num_tracks']}")

    print("\n" + "=" * 80)
    print("COLD START INGESTION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Clip File':28s} | {'Duration (s)':12s} | {'Wall-Clock (s)':14s} | {'Ratio (Wall/Dur)':16s} | {'Tracks':6s} | {'Frames':6s}")
    print("-" * 95)
    for r in results:
        print(f"{r['clip']:28s} | {r['duration_s']:12.2f} | {r['wall_time_s']:14.2f} | {r['ratio']:15.2f}x | {r['tracks']:6d} | {r['frames']:6d}")

    total_ratio = total_wall_time / total_video_dur if total_video_dur > 0 else 0.0
    print("-" * 95)
    print(f"{'TOTAL / OVERALL':28s} | {total_video_dur:12.2f} | {total_wall_time:14.2f} | {total_ratio:15.2f}x")

if __name__ == "__main__":
    main()
