import os
import sys
import time
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingest.detector import GroundingDinoDetector
from src.ingest.pipeline import IngestPipeline
from src.query.search import SearchEngine


def main():
    db_path = ROOT_DIR / "index_mobile" / "index.db"
    snapshots_dir = ROOT_DIR / "index_mobile" / "snapshots"
    video_path = ROOT_DIR / "footage" / "recorded" / "mobile_cam03_20261009_032158.mp4"

    print("=" * 80)
    print("RE-PROCESSING MOBILE CAM 03: PERFECT OBJECT INGEST PIPELINE")
    print(f"Video: {video_path}")
    print(f"Database: {db_path}")
    print("=" * 80)

    # 1. Clean out the noisy 1-hit live stream tracks for mobile_cam03
    conn = sqlite3.connect(db_path)
    with conn:
        deleted = conn.execute("DELETE FROM tracks WHERE camera = 'mobile_cam03';").rowcount
        print(f"Cleaned {deleted} noisy 1-hit live tracks for mobile_cam03 from database.")

    # 2. Initialize GroundingDinoDetector with full desk vocabulary
    print("\n[1/2] Loading GroundingDinoDetector with fine-grained desk vocabulary...")
    detector = GroundingDinoDetector(
        vocab_path=str(ROOT_DIR / "config" / "test_vocab.yaml"),
        device="cuda:0",
        conf_thresh=0.25,
        text_thresh=0.25
    )

    # 3. Ingest with strict quality and hit filtering
    print("\n[2/2] Ingesting mobile_cam03 with track quality & stability filtering...")
    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        device="cuda:0",
        custom_detector=detector
    )

    res = pipeline.ingest_video(
        video_path=video_path,
        manual_camera="mobile_cam03",
        stride=5,
        conf_thresh=0.25,
        min_duration_s=1.0,
        min_hits=3,
        min_mean_conf=0.30,
        force=True
    )

    print("\n" + "=" * 80)
    print("INGEST COMPLETE")
    print(f"Status: {res['status']}")
    print(f"Tracks indexed: {res.get('tracks_indexed', 0)}")
    print("=" * 80)

    # Print all new confirmed tracks
    cur = conn.cursor()
    print("\nConfirmed Stored Tracks for mobile_cam03:")
    for row in cur.execute("SELECT id, label, t_start, t_end, offset_best, hits, quality, snapshot FROM tracks WHERE camera = 'mobile_cam03'").fetchall():
        print(f"  Track: {row[0]} | Label: {row[1]} | Time: {row[4]:.1f}s | Hits: {row[5]} | Quality: {row[6]:.2f} | Snapshot: {row[7]}")

    # 4. Test natural language query
    print("\n" + "=" * 80)
    print("VERIFYING RETRIEVAL FOR: 'where is the juice box?'")
    print("=" * 80)
    engine = SearchEngine(db_path=db_path)
    q_res = engine.query("where is the juice box?", top_k=3)
    for i, r in enumerate(q_res.get("results", []), 1):
        p = Path(r.snapshot_path)
        print(f"  [{i}] ID: {r.result_id} | Label: {r.label} | Camera: {r.camera} | Offset: {r.offset_seconds:.1f}s")
        print(f"      Snapshot: {r.snapshot_path} (exists={p.exists()})")


if __name__ == "__main__":
    main()
