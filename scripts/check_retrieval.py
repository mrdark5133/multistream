import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query.search import SearchEngine
from src.ingest.embed import SigLIPEmbedder

def main():
    embedder = SigLIPEmbedder(device="cuda:0")
    queries = ["a red car", "a person with a bag", "a bus", "a person"]

    for db_name in ["index_base", "index_phase1b"]:
        print("=" * 80)
        print(f"RETRIEVAL CHECK ON {db_name}")
        print("=" * 80)
        engine = SearchEngine(db_path=f"{db_name}/index.db", embedder=embedder)
        for q in queries:
            print(f"\n--- Query: \"{q}\" ---")
            _, results = engine.search(q, top_k=3)
            for rank, r in enumerate(results[:3], 1):
                print(f"  Rank {rank}: ID={r.result_id} | Score={r.score:.4f} | Label={r.label} | Color={r.color} | Snapshot={r.snapshot_path}")

if __name__ == "__main__":
    main()
