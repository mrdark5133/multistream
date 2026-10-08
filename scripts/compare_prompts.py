import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query.search import SearchEngine

def main():
    db_path = Path("index_base/index.db")
    engine = SearchEngine(db_path=db_path, device="cuda:0")

    variants = [
        "red car",
        "a photo of a red car",
        "a car"
    ]

    print("=" * 90)
    print("PROMPT VARIANT COMPARISON (TOP-3 RESULTS PER VARIANT)")
    print("=" * 90)

    for query_str in variants:
        parsed, results = engine.search(query_str, top_k=3)
        print(f"\nQuery: '{query_str}' -> Parsed Object Prompt: '{parsed.object_prompt}'")
        print("-" * 90)
        for i, r in enumerate(results[:3]):
            print(f"  Rank {i+1}: ID={r.result_id:<25} Cam={r.camera:<16} Label={r.label:<8} Timestamp={r.timestamp:<26} Offset={r.offset_seconds:<6.3f}s Score={r.score:.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
