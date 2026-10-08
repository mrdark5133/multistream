import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
from src.query.search import SearchEngine
from scripts.eval import is_hit

def run_analysis():
    engine = SearchEngine(db_path="index_base/index.db", device="cuda:0")
    with open("eval/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    failures = []
    
    for split in ["dev", "held-out"]:
        print(f"\n{'='*40} Split: {split} {'='*40}")
        sub_q = [q for q in queries if q.get("split") == split]
        for q in sub_q:
            res = engine.query(q["query"], top_k=5)
            results_list = res.get("results", [])
            hits = [is_hit(r, q["truth"]) for r in results_list]
            ranks = [i+1 for i, (hit, _) in enumerate(hits) if hit]
            hit_rank = ranks[0] if ranks else 0
            
            if hit_rank != 1:
                fail_info = {
                    "id": q["id"],
                    "split": split,
                    "query": q["query"],
                    "truth": q["truth"],
                    "hit_rank": hit_rank,
                    "top_candidates": []
                }
                print(f"\n[SUBOPTIMAL / FAILURE] ID: {q['id']} ({split}) | \"{q['query']}\"")
                print(f"  Truth: {q['truth']}")
                print(f"  First Hit Rank: {hit_rank} (Top-5 Hit: {hit_rank > 0})")
                for idx, (r, (hit, err)) in enumerate(zip(results_list, hits), 1):
                    hit_str = "HIT" if hit else "MISS"
                    print(f"    Rank {idx} [{hit_str}]: score={r.score:.4f} cam={r.camera} time={r.timestamp} label={r.label} id={r.result_id}")
                    print(f"      Snapshot: {r.snapshot_path} | Time Err: {err:.2f}s")
                    fail_info["top_candidates"].append({
                        "rank": idx, "is_hit": hit, "score": r.score, "camera": r.camera,
                        "timestamp": r.timestamp, "label": r.label, "id": r.result_id, "snapshot": r.snapshot_path
                    })
                failures.append(fail_info)
            else:
                top_hit = results_list[0]
                print(f"[SUCCESS R@1] ID: {q['id']} ({split}) | \"{q['query']}\" -> Rank 1: score={top_hit.score:.4f} label={top_hit.label} id={top_hit.result_id}")

    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY OF SUBOPTIMAL / FAILURE CASES: {len(failures)} total")
    for f in failures:
        print(f"- {f['id']} ({f['split']}): rank={f['hit_rank']} | \"{f['query']}\"")

def test_camera_fix():
    print("\n" + "="*80)
    print("TESTING STRICT CAMERA FILTERING ON test_01 ('a white car passing at cam_landscape')")
    print("="*80)
    import sqlite3
    import numpy as np
    from src.ingest.embed import SigLIPEmbedder
    from src.index.db import blob_to_emb
    from src.query.search import SearchResult

    conn = sqlite3.connect("index_base/index.db")
    conn.row_factory = sqlite3.Row
    embedder = SigLIPEmbedder(device="cuda:0")
    
    # Try both prompts
    for p in ["a white car", "a photo of a white car"]:
        q_emb = embedder.embed_text([p])[0]
        # Query cam_landscape strictly
        rows = conn.execute(
            "SELECT * FROM tracks WHERE LOWER(camera) IN (?, ?)",
            ("cam_landscape", "landscape")
        ).fetchall()
        scored = []
        for r in rows:
            emb = blob_to_emb(r["emb"])
            score = float(np.dot(q_emb, emb))
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)

        truth = [{"camera": "cam_landscape", "start": "2026-10-08T18:00:00", "end": "2026-10-08T18:00:15", "label": "car"}]
        print(f"\nPrompt: '{p}' (Filtered strictly to cam_landscape):")
        for idx, (s, r) in enumerate(scored[:5], 1):
            res = SearchResult(
                result_id=r["id"], result_type="track", camera=r["camera"],
                timestamp=r["t_best"], offset_seconds=float(r["offset_best"]),
                score=s, label=r["label"], video_path=r["video"], snapshot_path=r["snapshot"]
            )
            hit, err = is_hit(res, truth)
            hit_str = "HIT" if hit else "MISS"
            print(f"  Rank {idx} [{hit_str}]: score={s:.4f} label={r['label']} time={r['t_best']} id={r['id']} (err={err:.2f}s)")

def test_landscape2_tracks():
    print("\n" + "="*80)
    print("INSPECTING TRACKS AT cam_landscape2 FOR test_02 and test_06")
    print("="*80)
    import sqlite3
    conn = sqlite3.connect("index_base/index.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, track_id, label, t_start, t_end, t_best, offset_best, snapshot FROM tracks WHERE LOWER(camera)='cam_landscape2' ORDER BY t_best").fetchall()
    print(f"Total tracks at cam_landscape2: {len(rows)}")
    for r in rows:
        print(f"  {r['id']:<24} label={r['label']:<12} start={r['t_start']} end={r['t_end']} best={r['t_best']} snap={r['snapshot']}")

def test_prompts_and_labels():
    print("\n" + "="*80)
    print("TESTING PROMPT VARIANTS & DETECTOR LABEL ALIGNMENT AT cam_landscape2")
    print("="*80)
    import sqlite3
    import numpy as np
    from src.ingest.embed import SigLIPEmbedder
    from src.index.db import blob_to_emb

    conn = sqlite3.connect("index_base/index.db")
    conn.row_factory = sqlite3.Row
    embedder = SigLIPEmbedder(device="cuda:0")

    rows = conn.execute("SELECT * FROM tracks WHERE LOWER(camera)='cam_landscape2'").fetchall()

    for p in ["utility cart", "a utility cart", "a photo of a utility cart"]:
        q_emb = embedder.embed_text([p])[0]
        scored = []
        for r in rows:
            emb = blob_to_emb(r["emb"])
            s = float(np.dot(q_emb, emb))
            scored.append((s, r["id"], r["label"], r["t_best"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"\nPrompt: '{p}'")
        for rank, (s, tid, lbl, t) in enumerate(scored[:5], 1):
            print(f"  Rank {rank}: score={s:.4f} id={tid:<24} label={lbl:<12} time={t}")

    print("\n--- Testing Attribute / Label Boost (Task 6.4) ---")
    # What if a track whose open-vocab detector label matches the query keyword gets a modest boost (e.g. +0.02)?
    for p, keyword in [("a utility cart", "utility cart"), ("a large truck", "truck")]:
        q_emb = embedder.embed_text([f"a photo of {p}"])[0]
        scored = []
        for r in rows:
            emb = blob_to_emb(r["emb"])
            raw_s = float(np.dot(q_emb, emb))
            boost = 0.025 if keyword in r["label"].lower() or r["label"].lower() in keyword else 0.0
            final_s = raw_s + boost
            scored.append((final_s, raw_s, boost, r["id"], r["label"], r["t_best"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"\nQuery: '{p}' with Detector Label Alignment (+0.025 if label match):")
        for rank, (fs, rs, b, tid, lbl, t) in enumerate(scored[:5], 1):
            print(f"  Rank {rank}: final={fs:.4f} (raw={rs:.4f}, boost={b:.3f}) id={tid:<24} label={lbl:<12} time={t}")

if __name__ == "__main__":
    run_analysis()
    test_camera_fix()
    test_prompts_and_labels()
