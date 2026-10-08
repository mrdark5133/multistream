import json
import sqlite3
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.embed import SigLIPEmbedder
from src.index.db import get_db, blob_to_emb

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

QUERIES = [
    "a red car",
    "a person with a bag",
    "a bus",
    "a person"
]

def search_index(db_path: str, query: str, embedder: SigLIPEmbedder, restrict_dev: bool = False, top_k: int = 3):
    conn = get_db(db_path)
    q_emb = embedder.embed_text(query).flatten()

    cols = [col[1] for col in conn.execute("PRAGMA table_info(tracks)").fetchall()]
    grp_col = '"group"' if 'group' in cols else "'unknown'"

    if restrict_dev:
        placeholders = ",".join(["?"] * len(DEV_CLIPS))
        cur = conn.execute(
            f"SELECT id, video, label, {grp_col} as grp, snapshot, emb FROM tracks WHERE video IN ({placeholders})",
            DEV_CLIPS
        )
    else:
        cur = conn.execute(f"SELECT id, video, label, {grp_col} as grp, snapshot, emb FROM tracks")

    rows = cur.fetchall()
    results = []
    for r in rows:
        t_emb = blob_to_emb(r["emb"])
        score = float(np.dot(q_emb, t_emb))
        results.append({
            "id": r["id"],
            "video": r["video"],
            "label": r["label"],
            "group": r["grp"],
            "snapshot": r["snapshot"],
            "score": round(score, 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    conn.close()
    return results[:top_k], len(rows)

def inspect_bus_tracks(db_path: str):
    conn = get_db(db_path)
    cols = [col[1] for col in conn.execute("PRAGMA table_info(tracks)").fetchall()]
    grp_col = '"group"' if 'group' in cols else ('group_name' if 'group_name' in cols else "'unknown'")
    cur = conn.execute(f"SELECT id, video, label, {grp_col} as grp, snapshot FROM tracks WHERE label = 'bus' OR {grp_col} = 'bus'")
    bus_tracks = [dict(r) for r in cur.fetchall()]
    cur2 = conn.execute("SELECT DISTINCT label, count(*) as cnt FROM tracks GROUP BY label")
    class_counts = {r["label"]: r["cnt"] for r in cur2.fetchall()}
    conn.close()
    return bus_tracks, class_counts

def main():
    embedder = SigLIPEmbedder(device="cuda:0")

    print("=" * 80)
    print("DEV QUERIES RETRIEVAL COMPARISON")
    print("=" * 80)

    for q in QUERIES:
        print(f"\nQUERY: '{q}'")

        # index_base restricted to 5 dev clips
        base_top, total_base = search_index("index_base/index.db", q, embedder, restrict_dev=True, top_k=3)
        print(f"  [index_base (5 dev clips only, total tracks = {total_base})]")
        for rank, res in enumerate(base_top, 1):
            print(f"    {rank}. id: {res['id']:25s} | label: {res['label']:8s} | score: {res['score']:.4f} | snapshot: {res['snapshot']}")

        # index_phase1b (contains only the 5 dev clips)
        p1b_top, total_p1b = search_index("index_phase1b/index.db", q, embedder, restrict_dev=False, top_k=3)
        print(f"  [index_phase1b (5 dev clips, total tracks = {total_p1b})]")
        for rank, res in enumerate(p1b_top, 1):
            print(f"    {rank}. id: {res['id']:25s} | label: {res['label']:8s} | score: {res['score']:.4f} | snapshot: {res['snapshot']}")

    print("\n" + "=" * 80)
    print("BUS TRACKS AUDIT")
    print("=" * 80)
    p1b_bus, p1b_classes = inspect_bus_tracks("index_phase1b/index.db")
    print(f"Total 'bus' label tracks in index_phase1b: {len(p1b_bus)}")
    for b in p1b_bus:
        print("  ", b)
    print("\nClass distribution in index_phase1b tracks:")
    for k, v in sorted(p1b_classes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:15s}: {v}")

    base_bus, base_classes = inspect_bus_tracks("index_base/index.db")
    print(f"\nTotal 'bus' label tracks in index_base (all clips): {len(base_bus)}")
    for b in base_bus:
        print("  ", b)

if __name__ == "__main__":
    main()
