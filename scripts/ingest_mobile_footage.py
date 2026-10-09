import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingest.detector import GroundingDinoDetector
from src.ingest.pipeline import IngestPipeline
from src.query.search import SearchEngine


def main():
    video_path = ROOT_DIR / "footage" / "mobile_cam02_20261009_024019.mp4"
    if not video_path.exists():
        video_path = ROOT_DIR / "footage" / "recorded" / "mobile_cam02_20261009_024019.mp4"

    db_path = ROOT_DIR / "index_mobile" / "index.db"
    snapshots_dir = ROOT_DIR / "index_mobile" / "snapshots"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INGESTING MOBILE FOOTAGE WITH GROUNDING-DINO OPEN-VOCABULARY DETECTOR")
    print(f"Video: {video_path}")
    print(f"Database: {db_path}")
    print(f"Snapshots: {snapshots_dir}")
    print("=" * 80)

    t0 = time.perf_counter()

    print("\n[1/3] Initializing GroundingDinoDetector...")
    detector = GroundingDinoDetector(
        vocab_path=str(ROOT_DIR / "config" / "test_vocab.yaml"),
        device="cuda:0",
        conf_thresh=0.20,
        text_thresh=0.20
    )
    print("Detector initialized successfully.")

    print("\n[2/3] Setting up IngestPipeline with Grounding-DINO + SigLIP...")
    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        device="cuda:0",
        custom_detector=detector
    )

    print("\n[3/3] Running ingest on mobile footage (stride=5)...")
    res = pipeline.ingest_video(
        video_path=video_path,
        manual_camera="mobile_cam02",
        stride=5,
        conf_thresh=0.20,
        min_duration_s=1.0,
        min_hits=3,
        min_mean_conf=0.30,
        force=True
    )

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"INGEST COMPLETE in {elapsed:.1f} s")
    print(f"Status: {res['status']}")
    print(f"Tracks indexed: {res.get('tracks_indexed', 0)}")
    print(f"Frames indexed: {res.get('frames_indexed', 0)}")
    print("=" * 80)

    # Verify Search and Evidence Retrieval
    print("\n" + "=" * 80)
    print("VERIFYING NATURAL LANGUAGE QUERIES WITH EVIDENCE")
    print("=" * 80)

    engine = SearchEngine(db_path=db_path)

    test_queries = [
        "where is the juice box?",
        "juice box",
        "a bluetooth speaker",
        "a phone charger",
        "a soda can"
    ]

    for q in test_queries:
        print(f"\n--- Query: '{q}' ---")
        q_res = engine.query(q, top_k=3)
        print(f"Status: {q_res['status']}")
        results = q_res.get("results", [])
        if not results:
            print("No matching results found.")
            continue

        for i, r in enumerate(results, 1):
            snap_path = ROOT_DIR / "index_mobile" / "snapshots" / "mobile_cam02" / f"{r.result_id}.jpg"
            snap_exists = snap_path.exists()
            print(f"  [{i}] ID: {r.result_id} | Label: {r.label} | Score: {r.score:.3f} | Offset: {r.offset_seconds:.2f}s | Timestamp: {r.timestamp}")
            print(f"      Color: {r.color} | Snapshot exists: {snap_exists} ({snap_path})")


if __name__ == "__main__":
    main()
