import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.index.db import get_db, list_videos
from src.query.search import SearchEngine
from src.query.reid import find_all_cross_camera_associations
from src.alerts.triggers import list_alert_rules, get_alert_events


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print_header("MULTIStream END-TO-END DEMO WALKTHROUGH")
    print("A Multi-Camera Video Retrieval & Spatio-Temporal Intelligence Engine")
    print(f"Root Directory: {ROOT_DIR}")

    db_path = ROOT_DIR / "index_base" / "index.db"
    if not db_path.exists():
        print(f"Error: Database {db_path} does not exist. Run ingest first.")
        sys.exit(1)

    conn = get_db(db_path)

    # 1. Database & Index Status
    print_header("STEP 1: Indexed Footage & Camera Topology")
    videos = list_videos(conn)
    cur = conn.execute("SELECT COUNT(*) as trk_count FROM tracks;")
    num_tracks = cur.fetchone()["trk_count"]
    cur = conn.execute("SELECT COUNT(*) as frm_count FROM frames;")
    num_frames = cur.fetchone()["frm_count"]

    print(f"Total Videos Indexed:  {len(videos)}")
    print(f"Total Object Tracks:   {num_tracks}")
    print(f"Total Sampled Frames:  {num_frames}")
    print("\nIndexed Cameras:")
    for v in videos:
        print(f"  - Camera: {v['camera']:<16} | File: {Path(v['path']).name:<32} | Start: {v['start_time']} ({v['start_source']})")

    # 2. Vector Search & Spatio-Temporal Retrieval
    print_header("STEP 2: Natural Language Query & Spatio-Temporal Search")
    test_query = "pedestrian on cam_landscape"
    print(f"Executing Query: \"{test_query}\"")

    engine = SearchEngine(db_path=db_path)
    t0 = time.perf_counter()
    parsed, results = engine.search(query_text=test_query, top_k=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Parsed Object:   {parsed.object_prompt}")
    print(f"Parsed Camera:   {parsed.resolved_camera or parsed.location}")
    print(f"Query Latency:   {elapsed_ms:.2f} ms")
    print(f"Results Found:   {len(results)}")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Camera: {r.camera:<14} | Score: {r.score:.4f} | Label: {r.label:<10} | Time: {r.timestamp}")
        print(f"      Snapshot: {r.snapshot_path}")

    # 3. Cross-Camera Track Re-identification
    print_header("STEP 3: Cross-Camera Track Re-identification (ReID)")
    print("Finding spatio-temporal entity hops across separate camera angles...")
    associations = find_all_cross_camera_associations(conn, min_similarity=0.85, top_k_per_track=2)
    print(f"Found {len(associations)} high-confidence cross-camera associations (Cosine Sim >= 0.85):")
    for a in associations[:3]:
        print(f"  - Path: {a['path']}")
        print(f"    Sim:  {a['similarity']:.4f} | Delta: {a['time_delta_s']:+.1f}s")
        print(f"    ObjA: {a['track_a']} ({a['label_a']}) <-> ObjB: {a['track_b']} ({a['label_b']})")

    # 4. Standing Alert Rules & Event Triggers
    print_header("STEP 4: Standing Alert Rules & Real-Time Event Triggers")
    rules = list_alert_rules(conn, active_only=True)
    events = get_alert_events(conn, limit=5)
    print(f"Active Alert Rules:  {len(rules)}")
    for r in rules:
        print(f"  - Rule [{r['id']}]: '{r['rule_name']}' -> Query: \"{r['query_text']}\" (Cam: {r['camera_filter'] or 'ALL'})")

    print(f"\nTriggered Events Recorded: {len(events)}")
    for e in events[:3]:
        print(f"  - Event [{e['id']}] Rule: '{e['rule_name']}' | Cam: {e['camera']} | Score: {e['score']:.4f}")
        print(f"    Snapshot: {e.get('snapshot_path') or e.get('snapshot')}")

    # 5. REST API & Web GUI
    print_header("STEP 5: API & Interactive UI")
    print("FastAPI Web Server is active at: http://127.0.0.1:8000")
    print("Interactive Endpoints:")
    print("  - Web UI:           GET  http://127.0.0.1:8000/")
    print("  - Search API:       POST http://127.0.0.1:8000/api/query")
    print("  - Ingest API:       POST http://127.0.0.1:8000/api/ingest")
    print("  - Aliases API:      GET  http://127.0.0.1:8000/api/aliases")
    print("  - Videos Catalog:   GET  http://127.0.0.1:8000/api/videos")
    print("=" * 80)
    print("  MULTIStream Demo Walkthrough Completed Successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
