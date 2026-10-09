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

    print("=" * 80)
    print("CLEANING FAKE / CONTAMINATED OBJECTS & RE-INDEXING WITH GROUNDING-DINO")
    print(f"Database: {db_path}")
    print("=" * 80)

    # 1. Clean out all fake and contaminated tracks from mobile cameras
    conn = sqlite3.connect(db_path)
    with conn:
        deleted = conn.execute("DELETE FROM tracks WHERE camera IN ('mobile_cam01', 'mobile_cam02', 'mobile_cam03');").rowcount
        print(f"[1/4] Cleaned {deleted} fake/contaminated tracks from index_mobile database.")

    # 2. Initialize GroundingDinoDetector with strict desk vocabulary
    print("\n[2/4] Loading GroundingDinoDetector with strict canonical desk vocabulary...")
    detector = GroundingDinoDetector(
        vocab_path=str(ROOT_DIR / "config" / "test_vocab.yaml"),
        device="cuda:0",
        conf_thresh=0.20,
        text_thresh=0.20
    )
    print("Loaded detector classes:", detector.raw_classes)

    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        device="cuda:0",
        custom_detector=detector
    )

    # 3. Ingest mobile recordings
    videos_to_index = [
        ("mobile_cam02", ROOT_DIR / "footage" / "mobile_cam02_20261009_024019.mp4"),
        ("mobile_cam03", ROOT_DIR / "footage" / "recorded" / "mobile_cam03_20261009_032158.mp4")
    ]

    for cam_name, vpath in videos_to_index:
        if not vpath.exists():
            # Check alternative location in footage/recorded
            vpath_alt = ROOT_DIR / "footage" / "recorded" / vpath.name
            if vpath_alt.exists():
                vpath = vpath_alt

        print(f"\n[3/4] Ingesting {cam_name} from: {vpath.name}...")
        res = pipeline.ingest_video(
            video_path=vpath,
            manual_camera=cam_name,
            stride=5,
            conf_thresh=0.20,
            min_duration_s=0.3,
            min_hits=2,
            min_mean_conf=0.25,
            force=True
        )
        print(f"  Result: {res.get('status')} | Indexed: {res.get('tracks_indexed', 0)} tracks")

    # 4. Summary of confirmed tracks stored
    print("\n" + "=" * 80)
    print("CONFIRMED CLEAN STORED TRACKS IN DATABASE:")
    print("=" * 80)
    cur = conn.cursor()
    rows = cur.execute("SELECT id, camera, label, offset_start, offset_end, hits, quality, snapshot FROM tracks WHERE camera IN ('mobile_cam02', 'mobile_cam03') ORDER BY camera, offset_start").fetchall()
    for r in rows:
        print(f"  Camera: {r[1]} | Track: {r[0]} | Label: {r[2]} | Time: {r[3]:.1f}s - {r[4]:.1f}s | Hits: {r[5]} | Conf: {r[6]:.2f}")
        print(f"      Snapshot: {r[7]}")

    # 5. Verify queries
    print("\n" + "=" * 80)
    print("VERIFYING NATURAL LANGUAGE RETRIEVAL")
    print("=" * 80)
    engine = SearchEngine(db_path=db_path)
    engine.reload_parser()

    test_queries = [
        "where is the juice box?",
        "where is the extension board?",
        "a laptop",
        "a bluetooth speaker",
        "a phone charger"
    ]

    for q in test_queries:
        print(f"\n>>> Query: '{q}'")
        res = engine.query(q, top_k=2)
        for i, hit in enumerate(res.get("results", []), 1):
            snap_file = ROOT_DIR / hit.snapshot_path
            print(f"    [{i}] Camera: {hit.camera} | Label: {hit.label} | Score: {hit.similarity_score:.3f} | Offset: {hit.offset_seconds:.1f}s | SnapExists: {snap_file.exists()}")


if __name__ == "__main__":
    main()
