import pytest
from pathlib import Path
from src.query.search import SearchEngine


@pytest.fixture(scope="module")
def search_engine():
    db_path = Path("index_base/index.db")
    if not db_path.exists():
        pytest.skip("index_base/index.db does not exist")
    return SearchEngine(db_path=db_path)


def test_search_person(search_engine):
    parsed, results = search_engine.search("a person walking", top_k=5)
    assert len(results) > 0
    # Top result should have positive cosine similarity
    top = results[0]
    assert top.score > 0.0
    assert top.result_type in ("track", "frame")
    assert top.snapshot_path is not None


def test_search_car_landscape(search_engine):
    parsed, results = search_engine.search("a vehicle near cam_landscape", top_k=5)
    assert len(results) > 0
    # Every returned result must match the requested camera filter
    for r in results:
        assert "landscape" in r.camera.lower()


def test_temporal_deduplication(search_engine):
    # With small dedup window, returned results on same camera should not have identical timestamps
    parsed, results = search_engine.search("person", top_k=5, dedup_window_s=3.0)
    for i in range(len(results) - 1):
        r1, r2 = results[i], results[i + 1]
        if r1.camera == r2.camera:
            assert abs(r1.offset_seconds - r2.offset_seconds) >= 0.5
