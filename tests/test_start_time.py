import json
import pytest
from pathlib import Path
from src.ingest.start_time import (
    resolve_video_metadata,
    parse_filename_time,
    parse_ffprobe_rotation
)


def test_parse_filename_time():
    # Format: cam_YYYYMMDD_HHMMSS
    res = parse_filename_time("gate_cam_20261008_143000.mp4")
    assert res is not None
    iso, cam = res
    assert iso == "2026-10-08T14:30:00"
    assert "gate_cam" in cam

    # Format: YYYY-MM-DD_HH-MM-SS
    res = parse_filename_time("cctv1_2026-10-08_14-30-00.mp4")
    assert res is not None
    iso, cam = res
    assert iso == "2026-10-08T14:30:00"

    # No date in filename
    assert parse_filename_time("random_video.mp4") is None


def test_manual_override(tmp_path):
    video_file = tmp_path / "clip.mp4"
    video_file.touch()
    
    meta = resolve_video_metadata(
        video_file,
        manual_start_time="2026-10-08T12:00:00",
        manual_camera="front_yard"
    )
    assert meta.start_source == "manual"
    assert meta.start_time == "2026-10-08T12:00:00"
    assert meta.camera == "front_yard"


def test_manifest_resolution(tmp_path):
    video_file = tmp_path / "clip_a.mp4"
    video_file.touch()
    manifest_file = tmp_path / "manifest.json"
    
    manifest_data = {
        "clip_a.mp4": {
            "start_time": "2026-10-08T09:15:00",
            "camera": "cam_lobby"
        }
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    meta = resolve_video_metadata(video_file, manifest_path=manifest_file)
    assert meta.start_source == "manifest"
    assert meta.start_time == "2026-10-08T09:15:00"
    assert meta.camera == "cam_lobby"


def test_filename_fallback(tmp_path):
    video_file = tmp_path / "cam02_20261008_180000.mp4"
    video_file.touch()
    
    meta = resolve_video_metadata(video_file)
    assert meta.start_source == "filename"
    assert meta.start_time == "2026-10-08T18:00:00"
    assert meta.camera == "cam02"


def test_mtime_fallback(tmp_path):
    video_file = tmp_path / "unnamed_no_metadata.mp4"
    video_file.touch()
    
    meta = resolve_video_metadata(video_file)
    assert meta.start_source == "mtime_fallback"
    assert meta.camera == "unnamed_no_metadata"
    assert len(meta.start_time) > 10


def test_parse_rotation():
    probe_with_side = {
        "streams": [
            {
                "side_data_list": [{"rotation": -90}]
            }
        ]
    }
    assert parse_ffprobe_rotation(probe_with_side) == 270

    probe_with_tag = {
        "streams": [
            {
                "tags": {"rotate": "180"}
            }
        ]
    }
    assert parse_ffprobe_rotation(probe_with_tag) == 180

    probe_clean = {"streams": [{}]}
    assert parse_ffprobe_rotation(probe_clean) == 0
