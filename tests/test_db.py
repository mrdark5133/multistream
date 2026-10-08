import pytest
import sqlite3
import numpy as np
from pathlib import Path
from src.index.db import (
    init_db,
    set_config,
    get_config,
    emb_to_blob,
    blob_to_emb,
    insert_video,
    get_video,
    insert_track,
    insert_frames_batch
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_multistream.db"
    conn = init_db(db_file)
    yield conn
    conn.close()


def test_wal_mode_and_foreign_keys(temp_db):
    cur = temp_db.execute("PRAGMA journal_mode;")
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"
    
    cur = temp_db.execute("PRAGMA foreign_keys;")
    fk = cur.fetchone()[0]
    assert fk == 1


def test_config_table(temp_db):
    assert get_config(temp_db, "nonexistent") is None
    assert get_config(temp_db, "nonexistent", default="123") == "123"
    
    set_config(temp_db, "embedder", "google/siglip-base-patch16-224")
    assert get_config(temp_db, "embedder") == "google/siglip-base-patch16-224"
    
    # Overwrite
    set_config(temp_db, "embedder", "google/siglip-so400m-patch14-384")
    assert get_config(temp_db, "embedder") == "google/siglip-so400m-patch14-384"


def test_embedding_blob_roundtrip():
    original = np.random.randn(768).astype(np.float32)
    blob = emb_to_blob(original)
    recovered = blob_to_emb(blob)
    np.testing.assert_allclose(original, recovered, rtol=1e-6)


def test_video_and_track_fk(temp_db):
    # Track insert before video should fail due to foreign key
    dummy_emb = emb_to_blob(np.zeros(768, dtype=np.float32))
    track = {
        "id": "trk_01",
        "video": "footage/test_video01.mp4",
        "camera": "cam_gate",
        "track_id": 1,
        "label": "person",
        "t_start": "2026-10-08T10:00:00",
        "t_end": "2026-10-08T10:00:02",
        "t_best": "2026-10-08T10:00:01",
        "offset_start": 0.0,
        "offset_end": 2.0,
        "bbox_px": "[10, 10, 50, 100]",
        "bbox_norm": "[0.1, 0.1, 0.5, 0.8]",
        "snapshot": "snapshots/trk_01.jpg",
        "emb": dummy_emb
    }
    with pytest.raises(sqlite3.IntegrityError):
        insert_track(temp_db, track)
        
    # Insert video first
    insert_video(
        temp_db,
        path="footage/test_video01.mp4",
        camera="cam_gate",
        start_time="2026-10-08T10:00:00",
        fps=30.0,
        width=478,
        height=850,
        duration_s=4.26,
        start_source="manifest",
        indexed_at="2026-10-08T12:00:00",
        rotation=0
    )
    v = get_video(temp_db, "footage/test_video01.mp4")
    assert v is not None
    assert v["camera"] == "cam_gate"
    assert v["rotation"] == 0
    
    # Now track insert succeeds
    insert_track(temp_db, track)
    cur = temp_db.execute("SELECT * FROM tracks WHERE id = 'trk_01';")
    row = cur.fetchone()
    assert row is not None
    assert row["label"] == "person"
