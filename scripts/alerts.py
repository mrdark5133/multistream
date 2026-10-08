import argparse
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.index.db import get_db
from src.alerts.triggers import (
    create_alert_rule,
    list_alert_rules,
    delete_alert_rule,
    evaluate_alert_rules,
    get_alert_events
)
from src.ingest.embed import SigLIPEmbedder


def main():
    parser = argparse.ArgumentParser(description="MULTIStream Standing Queries & Event Alert Manager")
    parser.add_argument("--db", type=str, default="index_base/index.db", help="Path to SQLite database")

    # Actions
    parser.add_argument("--add", action="store_true", help="Add new standing alert rule")
    parser.add_argument("--name", type=str, help="Human-readable rule name")
    parser.add_argument("--query", type=str, help="Natural language query description")
    parser.add_argument("--camera", type=str, default=None, help="Camera filter (optional)")
    parser.add_argument("--min-score", type=float, default=0.05, help="Minimum matching score threshold (default 0.05)")

    parser.add_argument("--list-rules", action="store_true", help="List all standing alert rules")
    parser.add_argument("--delete", type=str, help="Delete an alert rule by ID")

    parser.add_argument("--evaluate", action="store_true", help="Evaluate all active alert rules against database tracks")
    parser.add_argument("--list-events", action="store_true", help="List triggered alert events")
    parser.add_argument("--rule-id", type=str, default=None, help="Filter events by rule ID")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = get_db(db_path)

    if args.add:
        if not args.name or not args.query:
            print("Error: --name and --query are required when adding a rule.", file=sys.stderr)
            sys.exit(1)
        r_id = create_alert_rule(
            conn=conn,
            rule_name=args.name,
            query_text=args.query,
            camera_filter=args.camera,
            min_score=args.min_score
        )
        print(f"Created alert rule: {r_id} | Name: '{args.name}' | Query: '{args.query}' | Min Score: {args.min_score}")
        return

    if args.delete:
        ok = delete_alert_rule(conn, args.delete)
        if ok:
            print(f"Successfully deleted alert rule {args.delete} and associated events.")
        else:
            print(f"Rule ID {args.delete} not found.", file=sys.stderr)
        return

    if args.list_rules:
        rules = list_alert_rules(conn, active_only=False)
        if args.json:
            print(json.dumps(rules, indent=2))
            return
        print("=" * 80)
        print(f"STANDING ALERT RULES ({len(rules)} total)")
        print("=" * 80)
        for r in rules:
            status = "ACTIVE" if r["is_active"] else "INACTIVE"
            cam = r["camera_filter"] or "ALL CAMERAS"
            last = r["last_triggered_at"] or "never"
            print(f"[{r['id']}] [{status}] {r['rule_name']}")
            print(f"    Query:      \"{r['query_text']}\"")
            print(f"    Camera:     {cam} | Min Score: {r['min_score']}")
            print(f"    Created:    {r['created_at']} | Last Triggered: {last}")
            print("-" * 80)
        return

    if args.evaluate:
        rules = list_alert_rules(conn, active_only=True)
        if not rules:
            print("No active standing alert rules found. Add one with --add --name ... --query ...")
            return

        print(f"Loading embedder to evaluate {len(rules)} active rule(s)...")
        embedder = SigLIPEmbedder(model_id="google/siglip-base-patch16-224", device="cuda:0")
        events = evaluate_alert_rules(conn=conn, embedder=embedder, rules=rules)

        if args.json:
            print(json.dumps(events, indent=2))
            return

        print("=" * 80)
        print(f"ALERT EVALUATION RUN COMPLETE: {len(events)} new trigger(s)")
        print("=" * 80)
        for e in events:
            print(f"Trigger [{e['event_id']}] Rule: '{e['rule_name']}'")
            print(f"  Track:     {e['track_id']} (Cam: {e['camera']}, Label: {e['label']})")
            print(f"  Score:     {e['score']:.4f} (Threshold: matching)")
            print(f"  Timestamp: {e['timestamp']}")
            print(f"  Snapshot:  {e['snapshot']}")
            print("-" * 80)
        return

    if args.list_events:
        events = get_alert_events(conn, rule_id=args.rule_id)
        if args.json:
            print(json.dumps(events, indent=2))
            return
        print("=" * 80)
        print(f"TRIGGERED ALERT EVENTS ({len(events)} total)")
        print("=" * 80)
        for e in events:
            print(f"[{e['id']}] Rule: '{e['rule_name']}' ({e['query_text']})")
            print(f"  Track:     {e['track_id']} | Cam: {e['camera']}")
            snap = e.get("snapshot_path") or e.get("snapshot")
            print(f"  Snapshot:  {snap}")
            print("-" * 80)
        return

    # If no flag provided, display help
    parser.print_help()


if __name__ == "__main__":
    main()
