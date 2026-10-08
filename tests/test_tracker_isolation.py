import pytest
import numpy as np
from src.ingest.detect_track import load_yolo_world, reset_tracker, process_video_tracks


@pytest.fixture(scope="module")
def yolo_and_classes():
    return load_yolo_world(device="cuda:0")


def test_reset_tracker_resets_basetrack_and_active_tracks(yolo_and_classes):
    yolo, classes = yolo_and_classes
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run tracking to populate predictor tracker
    yolo.track(dummy, persist=True, tracker="bytetrack.yaml", verbose=False)
    assert hasattr(yolo.predictor, "trackers")
    assert len(yolo.predictor.trackers) > 0
    
    # Simulate non-zero track counter
    from ultralytics.trackers.basetrack import BaseTrack
    BaseTrack._count = 42
    
    # Now reset
    reset_tracker(yolo)
    assert BaseTrack._count == 0
    assert len(yolo.predictor.trackers[0].tracked_stracks) == 0


def test_multi_video_tracker_isolation(yolo_and_classes):
    yolo, classes = yolo_and_classes
    
    # 1. Run video 2 standalone to obtain clean baseline track IDs
    tracks_standalone, _ = process_video_tracks(
        video_path="footage/test_video02.mp4",
        yolo_model=yolo,
        classes=classes,
        start_time_iso="2026-10-08T10:00:10",
        fps=30.0,
        duration_s=9.13,
        stride=5,
        conf_thresh=0.25,
        imgsz=640
    )
    standalone_ids = [t.track_id for t in tracks_standalone]
    assert len(standalone_ids) > 0
    
    # 2. Run video 1 (generates tracks up to ID ~90)
    tracks1, _ = process_video_tracks(
        video_path="footage/test_video01.mp4",
        yolo_model=yolo,
        classes=classes,
        start_time_iso="2026-10-08T10:00:00",
        fps=30.0,
        duration_s=4.26,
        stride=5,
        conf_thresh=0.25,
        imgsz=640
    )
    assert len(tracks1) > 0
    assert max(t.track_id for t in tracks1) > 50
    
    # 3. Run video 2 again sequentially
    tracks2, _ = process_video_tracks(
        video_path="footage/test_video02.mp4",
        yolo_model=yolo,
        classes=classes,
        start_time_iso="2026-10-08T10:00:10",
        fps=30.0,
        duration_s=9.13,
        stride=5,
        conf_thresh=0.25,
        imgsz=640
    )
    sequential_ids = [t.track_id for t in tracks2]
    
    # Verification: sequential run track IDs must match standalone run exactly
    assert sequential_ids == standalone_ids, (
        f"Track state leaked across videos! Standalone: {standalone_ids}, Sequential: {sequential_ids}"
    )
