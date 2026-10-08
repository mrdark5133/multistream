import json
import sqlite3
import pytest
from pathlib import Path

from src.memory.aliases import AliasManager, normalize_alias_name, point_in_polygon
from src.query.search import SearchEngine


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary SQLite database initialized with MULTIStream schema."""
    db_file = tmp_path / "test_memory.db"
    conn = sqlite3.connect(db_file)
    schema_path = Path("src/index/schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    
    # Populate known cameras in videos table
    conn.execute(
        """
        INSERT INTO videos (path, camera, start_time, fps, width, height, duration_s, start_source, rotation, indexed_at)
        VALUES ('v1.mp4', 'cam_gate_2', '2026-10-08T18:00:00', 30.0, 1920, 1080, 60.0, 'manifest', 0, '2026-10-08T18:00:00'),
               ('v2.mp4', 'cam_landscape', '2026-10-08T18:00:00', 30.0, 1920, 1080, 60.0, 'manifest', 0, '2026-10-08T18:00:00');
        """
    )
    conn.commit()
    conn.close()
    return db_file


def test_alias_normalization():
    assert normalize_alias_name("The Main Gate!") == "main gate"
    assert normalize_alias_name("  the   Front Yard   ") == "front yard"
    assert normalize_alias_name("LOBBY") == "lobby"


def test_point_in_polygon_geometry():
    # Unit square: [0.0, 0.0] to [0.5, 0.5]
    poly = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    
    # Inside center
    assert point_in_polygon(0.25, 0.25, poly) is True
    # Outside
    assert point_in_polygon(0.75, 0.75, poly) is False
    assert point_in_polygon(0.6, 0.2, poly) is False
    assert point_in_polygon(0.2, 0.8, poly) is False
    # Empty polygon passes everything
    assert point_in_polygon(0.9, 0.9, []) is True


def test_unknown_referent_triggers_clarification(test_db_path):
    """Task 3.4.a: Unknown referent triggers clarification request with camera options."""
    engine = SearchEngine(db_path=test_db_path)
    
    resp = engine.query("a person at the main gate")
    assert resp["status"] == "clarify"
    assert resp["referent"] == "main gate"
    assert "cam_gate_2" in resp["options"]
    assert "cam_landscape" in resp["options"]


def test_stored_alias_resolves_without_asking_across_paraphrases(test_db_path):
    """Task 3.4.b: After saving, same and paraphrased queries resolve without asking."""
    engine = SearchEngine(db_path=test_db_path)
    
    # Save alias
    engine.save_alias("main gate", "cam_landscape")
    
    # Original phrasing
    resp1 = engine.query("a person at the main gate")
    assert resp1["status"] == "success"
    assert resp1["parsed"].location == "main gate"
    assert resp1["parsed"].resolved_camera == "cam_landscape"
    
    # Paraphrase 1: "near main gate"
    resp2 = engine.query("show cars near main gate")
    assert resp2["status"] == "success"
    assert resp2["parsed"].location == "main gate"
    assert resp2["parsed"].resolved_camera == "cam_landscape"

    # Paraphrase 2: "by the main gate"
    resp3 = engine.query("did a delivery truck pass by the main gate in the last hour?")
    assert resp3["status"] == "success"
    assert resp3["parsed"].location == "main gate"
    assert resp3["parsed"].resolved_camera == "cam_landscape"


def test_cross_process_restart_persistence(test_db_path):
    """Task 3.4.c: Restart test: save in one instance, load fresh instance, confirm resolution."""
    # Process 1: save alias
    engine1 = SearchEngine(db_path=test_db_path)
    engine1.save_alias("loading dock", "cam_landscape")
    del engine1  # Simulate process termination

    # Process 2: freshly loaded engine with fresh connection
    engine2 = SearchEngine(db_path=test_db_path)
    resp = engine2.query("a forklift at the loading dock")
    assert resp["status"] == "success"
    assert resp["parsed"].location == "loading dock"
    assert resp["parsed"].resolved_camera == "cam_landscape"


def test_known_camera_names_bypass_clarification(test_db_path):
    """Task 3.4.d: A referent that is an existing camera name bypasses clarification."""
    engine = SearchEngine(db_path=test_db_path)
    
    # Direct camera reference: cam_landscape
    resp = engine.query("a red car near cam_landscape")
    assert resp["status"] == "success"
    assert resp["parsed"].location == "cam_landscape"
    assert resp["parsed"].resolved_camera == "cam_landscape"


def test_polygon_spatial_filtering(test_db_path):
    """Task 3.4.e: Polygon filter retains only tracks whose bbox center is inside."""
    conn = sqlite3.connect(test_db_path)
    
    # Insert two tracks on cam_landscape:
    # Track A: center (0.2, 0.2) -> INSIDE quadrant [0.0, 0.5]
    # Track B: center (0.8, 0.8) -> OUTSIDE quadrant [0.0, 0.5]
    dummy_emb = (b"\x00" * 4) * 768
    conn.execute(
        """
        INSERT INTO tracks (id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb)
        VALUES ('trk_inside', 'v2.mp4', 'cam_landscape', 1, 'car', '2026-10-08T18:00:00', '2026-10-08T18:00:05', '2026-10-08T18:00:02', 0.0, 5.0, 2.0, '[100,100,300,300]', '[0.1, 0.1, 0.3, 0.3]', 's1.jpg', ?),
               ('trk_outside', 'v2.mp4', 'cam_landscape', 2, 'car', '2026-10-08T18:00:00', '2026-10-08T18:00:05', '2026-10-08T18:00:02', 0.0, 5.0, 2.0, '[700,700,900,900]', '[0.7, 0.7, 0.9, 0.9]', 's2.jpg', ?);
        """,
        (dummy_emb, dummy_emb)
    )
    conn.commit()
    conn.close()

    engine = SearchEngine(db_path=test_db_path)
    
    # Save alias with top-left quadrant polygon
    quadrant_poly = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    engine.save_alias("north parking", "cam_landscape", polygon_norm=quadrant_poly)

    parsed, results = engine.search("a car at north parking")
    result_ids = [r.result_id for r in results]
    assert "trk_inside" in result_ids
    assert "trk_outside" not in result_ids


def test_disallow_guessing_with_tempting_context(test_db_path):
    """Task 3.5: Unknown referent 'main gate' must NOT guess 'cam_gate_2' despite tempting lexical similarity."""
    engine = SearchEngine(db_path=test_db_path)
    resp = engine.query("a person at the main gate")
    assert resp["status"] == "clarify"
    assert resp["referent"] == "main gate"
    # Never guess cam_gate_2!
    assert resp.get("results") is None or len(resp.get("results", [])) == 0
