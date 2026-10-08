import argparse
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.index.db import get_db
from src.query.reid import find_cross_camera_matches, find_all_cross_camera_associations


def main():
    parser = argparse.ArgumentParser(description="MULTIStream Cross-Camera Track Re-identification")
    parser.add_argument("--db", type=str, default="index_base/index.db", help="Path to SQLite database")
    parser.add_argument("--track-id", type=str, help="Specific track ID to re-identify across cameras")
    parser.add_argument("--min-sim", type=float, default=0.65, help="Minimum cosine similarity threshold (default 0.65)")
    parser.add_argument("--max-gap-s", type=float, default=7200.0, help="Max temporal gap in seconds (default 7200s)")
    parser.add_argument("--top-k", type=int, default=5, help="Max matches to display")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = get_db(db_path)

    if args.track_id:
        try:
            result = find_cross_camera_matches(
                conn=conn,
                query_track_id=args.track_id,
                min_similarity=args.min_sim,
                max_time_gap_s=args.max_gap_s
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
            return

        qt = result["query_track"]
        print("=" * 70)
        print("CROSS-CAMERA TRACK RE-IDENTIFICATION")
        print("=" * 70)
        print(f"Query Track ID: {qt['id']}")
        print(f"Camera:         {qt['camera']}")
        print(f"Label:          {qt['label']}")
        print(f"Timestamp:      {qt['t_start']} to {qt['t_end']}")
        print(f"Snapshot:       {qt['snapshot']}")
        print("-" * 70)
        print(f"Candidate Trajectory: {result['trajectory_summary']}")
        print("-" * 70)
        print(f"Matches across other cameras (min_sim={args.min_sim}): {len(result['matches'])}")
        for i, m in enumerate(result["matches"][:args.top_k], 1):
            print(f"  [{i}] Cam: {m['camera']:<15} | Track: {m['track_id']} | Sim: {m['similarity']:.4f} | "
                  f"Time: {m['t_start']} ({m['time_delta_s']:+.1f}s {m['chronological_order']})")
            print(f"      Snapshot: {m['snapshot']}")
        print("=" * 70)

    else:
        # Scan all cross-camera candidate associations
        associations = find_all_cross_camera_associations(
            conn=conn,
            min_similarity=args.min_sim,
            max_time_gap_s=args.max_gap_s,
            top_k_per_track=args.top_k
        )

        if args.json:
            print(json.dumps(associations, indent=2))
            return

        print("=" * 80)
        print(f"CROSS-CAMERA TRACK ASSOCIATIONS (Found {len(associations)} pairs with sim >= {args.min_sim})")
        print("=" * 80)
        for i, assoc in enumerate(associations[:20], 1):
            print(f"[{i:02d}] {assoc['path']}")
            print(f"     Track A: {assoc['track_a']} ({assoc['camera_a']}, {assoc['label_a']})")
            print(f"     Track B: {assoc['track_b']} ({assoc['camera_b']}, {assoc['label_b']})")
            print(f"     Cosine Sim: {assoc['similarity']:.4f} | Temporal Delta: {assoc['time_delta_s']:+.1f}s")
            print(f"     Snapshots: {assoc['snapshot_a']} <-> {assoc['snapshot_b']}")
            print("-" * 80)


if __name__ == "__main__":
    main()
