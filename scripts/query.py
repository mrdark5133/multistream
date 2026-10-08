import argparse
import sys
import json
import time
import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query.search import SearchEngine
from src.query.clips import extract_video_clip


def main():
    parser = argparse.ArgumentParser(description="MULTIStream Natural Language Video Search CLI")
    parser.add_argument("query", type=str, nargs="?", default=None, help="Natural language query string")
    parser.add_argument("--query", "-q", type=str, dest="query_flag", default=None, help="Query string")
    parser.add_argument("--index-dir", type=str, default="index_base", help="Index storage directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--cut-clips", action="store_true", help="Extract video clips for top results")
    parser.add_argument("--clips-dir", type=str, default="clips", help="Output directory for clips")
    parser.add_argument("--save-alias", type=str, default=None, help="Save a human location alias name")
    parser.add_argument("--camera", type=str, default=None, help="Camera identifier for saved alias")
    parser.add_argument("--polygon", type=str, default=None, help="JSON string of normalized polygon coordinates [[x,y],...]")
    parser.add_argument("--now", type=str, default=None, help="Override reference timestamp ISO-8601")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    db_path = index_dir / "index.db"
    
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}. Please run ingest first.")
        sys.exit(1)

    engine = SearchEngine(db_path=db_path)

    # 1. Alias management
    if args.save_alias:
        if not args.camera:
            print("Error: --camera must be specified when using --save-alias.")
            sys.exit(1)
        poly = json.loads(args.polygon) if args.polygon else None
        engine.save_alias(args.save_alias, args.camera, polygon_norm=poly)
        if args.json:
            print(json.dumps({
                "status": "saved",
                "name": args.save_alias,
                "camera": args.camera,
                "polygon": poly
            }, indent=2))
        else:
            print(f"Successfully saved alias '{args.save_alias}' -> camera '{args.camera}' (polygon: {poly})")
        if not (args.query or args.query_flag):
            return

    q_text = args.query or args.query_flag
    if not q_text:
        parser.print_help()
        sys.exit(1)

    now_override = datetime.datetime.fromisoformat(args.now) if args.now else None

    t0 = time.perf_counter()
    resp = engine.query(
        query_text=q_text,
        top_k=args.top_k,
        now_override=now_override
    )
    t_search = time.perf_counter() - t0

    # 2. Handle clarification request
    if resp["status"] == "clarify":
        if args.json:
            out_clarify = {
                "status": "clarify",
                "referent": resp["referent"],
                "options": resp["options"]
            }
            print(json.dumps(out_clarify, indent=2))
        else:
            print("=" * 80)
            print("CLARIFICATION REQUIRED:")
            print(f"Location referent '{resp['referent']}' is unknown and not mapped to any camera.")
            print(f"Available cameras: {', '.join(resp['options'])}")
            print("To map this referent, run:")
            print(f"  python scripts/query.py --save-alias \"{resp['referent']}\" --camera <CAMERA_NAME>")
            print("=" * 80)
        return

    parsed = resp["parsed"]
    results = resp["results"]

    # Optional clip cutting
    if args.cut_clips:
        clips_dir = Path(args.clips_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if Path(r.video_path).exists():
                out_name = f"clip_{r.result_id}.mp4"
                clip_out = clips_dir / out_name
                cut_path = extract_video_clip(
                    video_path=r.video_path,
                    output_path=clip_out,
                    start_s=r.offset_seconds,
                    end_s=r.offset_seconds + 3.0,
                    padding_s=2.0
                )
                r.clip_path = cut_path

    if args.json:
        out_dict = {
            "query": q_text,
            "parsed": {
                "object_prompt": parsed.object_prompt,
                "location": parsed.location,
                "location_status": parsed.location_status,
                "resolved_camera": parsed.resolved_camera,
                "t_start": parsed.t_start,
                "t_end": parsed.t_end,
                "provider": parsed.provider
            },
            "latency_ms": round(t_search * 1000.0, 2),
            "results": [
                {
                    "rank": i + 1,
                    "id": r.result_id,
                    "type": r.result_type,
                    "camera": r.camera,
                    "timestamp": r.timestamp,
                    "offset_seconds": r.offset_seconds,
                    "score": r.score,
                    "label": r.label,
                    "video": r.video_path,
                    "snapshot": r.snapshot_path,
                    "clip": r.clip_path
                }
                for i, r in enumerate(results)
            ]
        }
        print(json.dumps(out_dict, indent=2))
        return

    print("=" * 80)
    print(f"MULTIStream Search: '{q_text}'")
    print(f"Parsed Object:   {parsed.object_prompt}")
    print(f"Parsed Location: {parsed.location or 'any'}")
    print(f"Parsed Time:     {parsed.t_start or 'any'} -> {parsed.t_end or 'any'}")
    print(f"Search Latency:  {t_search * 1000.0:.2f} ms")
    print("=" * 80)

    if not results:
        print("No matching results found above threshold.")
        return

    for i, r in enumerate(results):
        print(f"\n[Rank {i+1}] Score: {r.score:.4f} | Camera: {r.camera} | Time: {r.timestamp}")
        print(f"         Type: {r.result_type} | Label: {r.label} | Offset: {r.offset_seconds:.1f}s")
        print(f"         Snapshot: {r.snapshot_path}")
        if r.clip_path:
            print(f"         Clip:     {r.clip_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
