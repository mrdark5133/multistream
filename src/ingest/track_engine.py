import cv2
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace
import yaml

from src.utils.lab_color import extract_track_colors
from src.ingest.detector import get_object_group


@dataclass
class TrackObservation:
    frame_idx: int
    offset_s: float
    iso_time: str
    bbox_px: List[int]        # [x1, y1, x2, y2]
    bbox_norm: List[float]    # [x1/W, y1/H, x2/W, y2/H]
    conf: float
    quality: float            # conf * size * sharpness
    crop_bgr: np.ndarray


@dataclass
class TrackState:
    track_id: int
    observations: List[TrackObservation] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    best_quality: float = -1.0
    best_obs: Optional[TrackObservation] = None
    best_crop_bgr: Optional[np.ndarray] = None
    best_snapshot_bgr: Optional[np.ndarray] = None

    def add(self, obs: TrackObservation, label: str, conf: float, frame_bgr: Optional[np.ndarray] = None) -> None:
        self.observations.append(obs)
        self.labels.append(label)
        self.confs.append(conf)

        if obs.quality > self.best_quality or self.best_obs is None:
            self.best_quality = obs.quality
            self.best_obs = obs
            self.best_crop_bgr = obs.crop_bgr.copy()
            if frame_bgr is not None:
                # Store compact snapshot (~640px) to prevent multi-gigabyte RAM leaks on 4K video
                fh, fw = frame_bgr.shape[:2]
                scale = min(640 / max(fw, fh), 1.0)
                if scale < 1.0:
                    self.best_snapshot_bgr = cv2.resize(frame_bgr, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
                else:
                    self.best_snapshot_bgr = frame_bgr.copy()

    @property
    def hits(self) -> int:
        return len(self.observations)

    @property
    def duration_s(self) -> float:
        if not self.observations:
            return 0.0
        return max(0.0, self.observations[-1].offset_s - self.observations[0].offset_s)

    @property
    def mean_conf(self) -> float:
        if not self.confs:
            return 0.0
        return float(np.mean(self.confs))

    def get_voted_label(self) -> str:
        """Class voting weighted by confidence."""
        if not self.labels:
            return "unknown"
        weights: Dict[str, float] = {}
        for l, c in zip(self.labels, self.confs):
            weights[l] = weights.get(l, 0.0) + c
        return max(weights.items(), key=lambda x: x[1])[0]

    def get_best_observation(self) -> Tuple[TrackObservation, float]:
        """Return observation with highest crop quality."""
        if self.best_obs is not None:
            return self.best_obs, self.best_quality
        best_obs = max(self.observations, key=lambda o: o.quality)
        return best_obs, best_obs.quality


def pad_to_square(image: np.ndarray, pad_color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """Pad crop to 1:1 aspect ratio with neutral border to preserve aspect ratio."""
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if h == w:
        return image
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    pad_h_extra = (max_dim - h) % 2
    pad_w_extra = (max_dim - w) % 2

    padded = cv2.copyMakeBorder(
        image,
        pad_h, pad_h + pad_h_extra,
        pad_w, pad_w + pad_w_extra,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return padded


def compute_crop_quality(
    crop: np.ndarray,
    conf: float,
    bbox_px: List[int],
    frame_w: int,
    frame_h: int
) -> float:
    """
    Crop quality: Q = confidence * size * sharpness
    where:
    - size = sqrt((w*h) / (W*H)) normalized box dimension
    - sharpness = Laplacian variance normalized
    """
    bw = max(1, bbox_px[2] - bbox_px[0])
    bh = max(1, bbox_px[3] - bbox_px[1])
    size = np.sqrt((bw * bh) / max(1.0, float(frame_w * frame_h)))

    if crop is not None and crop.size > 0:
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Scale to reasonable range [0.05, 5.0]
        sharpness = float(np.clip(lap_var / 120.0, 0.05, 5.0))
    else:
        sharpness = 0.05

    return float(conf * size * sharpness)


def create_tracker(tracker_type: str = "bytetrack_tuned") -> Any:
    """Instantiate BYTETracker or BOTSORT with yaml config."""
    if tracker_type == "botsort":
        import ultralytics
        from pathlib import Path
        p = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
        cfg = yaml.safe_load(open(p))
        args = IterableSimpleNamespace(**cfg)
        return BOTSORT(args)
    else:
        cfg = yaml.safe_load(open("config/bytetrack_tuned.yaml"))
        args = IterableSimpleNamespace(**cfg)
        return BYTETracker(args)


def run_tracker_on_detections(
    tracker: Any,
    detections: List[Dict[str, Any]],
    frame_bgr: np.ndarray,
    frame_w: int,
    frame_h: int
) -> List[Dict[str, Any]]:
    """
    Feed detections to tracker and return updated tracked bounding boxes with track_ids.
    """
    if not detections:
        # Feed empty boxes to allow tracker internal state update
        dummy_boxes = Boxes(torch.zeros((0, 6)), (frame_h, frame_w))
        if isinstance(tracker, BOTSORT):
            tracker.update(dummy_boxes, img=frame_bgr)
        else:
            tracker.update(dummy_boxes)
        return []

    # Prepare tensor [x1, y1, x2, y2, conf, cls_idx]
    tensor_data = []
    cls_to_idx = {}
    idx_to_label = {}
    for d in detections:
        lbl = d["label"]
        if lbl not in cls_to_idx:
            idx = len(cls_to_idx)
            cls_to_idx[lbl] = idx
            idx_to_label[idx] = lbl
        c_idx = cls_to_idx[lbl]
        b = d["bbox"]
        tensor_data.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(d["conf"]), float(c_idx)])

    boxes = Boxes(torch.tensor(tensor_data, dtype=torch.float32), (frame_h, frame_w))
    if isinstance(tracker, BOTSORT):
        out = tracker.update(boxes, img=frame_bgr)
    else:
        out = tracker.update(boxes)

    tracked_results: List[Dict[str, Any]] = []
    if out is not None and len(out) > 0:
        for row in out:
            # row: [x1, y1, x2, y2, track_id, conf, cls, idx]
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tid = int(row[4])
            conf = float(row[5])
            c_idx = int(row[6])
            label = idx_to_label.get(c_idx, detections[0]["label"])
            tracked_results.append({
                "track_id": tid,
                "bbox": [max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)],
                "conf": conf,
                "label": label,
                "group": get_object_group(label)
            })

    return tracked_results


def filter_tracks(
    raw_tracks: Dict[int, TrackState],
    min_duration_s: float = 1.0,
    min_hits: int = 4,
    min_mean_conf: float = 0.35
) -> Tuple[List[TrackState], Dict[str, int]]:
    """
    Filter raw tracks according to criteria:
    - min_duration_s >= 1.0
    - min_hits >= 4
    - min_mean_conf >= 0.35
    Returns surviving tracks and drop count dictionary.
    """
    surviving: List[TrackState] = []
    dropped_counts = {
        "duration": 0,
        "hits": 0,
        "mean_conf": 0,
        "total_dropped": 0
    }

    for tid, st in raw_tracks.items():
        if st.duration_s < min_duration_s:
            dropped_counts["duration"] += 1
            dropped_counts["total_dropped"] += 1
            continue
        if st.hits < min_hits:
            dropped_counts["hits"] += 1
            dropped_counts["total_dropped"] += 1
            continue
        if st.mean_conf < min_mean_conf:
            dropped_counts["mean_conf"] += 1
            dropped_counts["total_dropped"] += 1
            continue
        surviving.append(st)

    # Sort chronologically by start time
    surviving.sort(key=lambda s: s.observations[0].offset_s)
    return surviving, dropped_counts


def stitch_tracks(
    tracks: List[TrackState],
    track_embs: List[np.ndarray],
    max_gap_s: float = 2.0,
    max_center_dist: float = 0.25,
    min_cosine_sim: float = 0.80
) -> Tuple[List[TrackState], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Merge same-group tracks with:
    - 0 <= gap <= max_gap_s
    - normalized center distance <= max_center_dist
    - embedding cosine similarity >= 0.88 for vehicle group, >= min_cosine_sim (0.80) for others
    - Strictly one-to-one merges: a track can have at most one successor and at most one predecessor.
    Returns merged tracks, updated embeddings, and list of merge details.
    """
    if len(tracks) <= 1:
        return tracks, track_embs, []

    # Find all valid candidate pairs
    candidate_pairs = []
    for i in range(len(tracks)):
        base_track = tracks[i]
        base_emb = track_embs[i]
        base_group = get_object_group(base_track.get_voted_label())
        required_thresh = 0.88 if base_group == "vehicle" else min_cosine_sim

        for j in range(i + 1, len(tracks)):
            cand_track = tracks[j]
            cand_group = get_object_group(cand_track.get_voted_label())
            if base_group != cand_group:
                continue

            gap_s = cand_track.observations[0].offset_s - base_track.observations[-1].offset_s
            if not (0.0 <= gap_s <= max_gap_s):
                continue

            b_last_box = base_track.observations[-1].bbox_norm
            c_first_box = cand_track.observations[0].bbox_norm
            bc_x = (b_last_box[0] + b_last_box[2]) / 2.0
            bc_y = (b_last_box[1] + b_last_box[3]) / 2.0
            cc_x = (c_first_box[0] + c_first_box[2]) / 2.0
            cc_y = (c_first_box[1] + c_first_box[3]) / 2.0
            center_dist = float(np.sqrt((bc_x - cc_x) ** 2 + (bc_y - cc_y) ** 2))
            if center_dist > max_center_dist:
                continue

            cand_emb = track_embs[j]
            cos_sim = float(np.dot(base_emb, cand_emb))
            if cos_sim < required_thresh:
                continue

            candidate_pairs.append({
                "i": i,
                "j": j,
                "base_id": base_track.track_id,
                "cand_id": cand_track.track_id,
                "label": base_track.get_voted_label(),
                "group": base_group,
                "gap_s": round(gap_s, 3),
                "center_dist": round(center_dist, 4),
                "cosine_sim": round(cos_sim, 4)
            })

    # Sort pairs by cosine similarity descending for greedy optimal matching
    candidate_pairs.sort(key=lambda x: x["cosine_sim"], reverse=True)

    has_successor = set()    # Base tracks that have already been assigned a successor
    has_predecessor = set()  # Candidate tracks that have already been assigned a predecessor
    selected_pairs = []

    for p in candidate_pairs:
        i, j = p["i"], p["j"]
        if i not in has_successor and j not in has_predecessor:
            has_successor.add(i)
            has_predecessor.add(j)
            selected_pairs.append(p)

    # Perform the merges
    merged_tracks_dict = {i: tracks[i] for i in range(len(tracks))}
    merged_embs_dict = {i: track_embs[i] for i in range(len(track_embs))}
    absorbed_indices = set()
    merge_logs = []

    for p in selected_pairs:
        i, j = p["i"], p["j"]
        base_track = merged_tracks_dict[i]
        cand_track = merged_tracks_dict[j]
        base_emb = merged_embs_dict[i]
        cand_emb = merged_embs_dict[j]

        # Merge observations
        base_track.observations.extend(cand_track.observations)
        base_track.labels.extend(cand_track.labels)
        base_track.confs.extend(cand_track.confs)

        # Average and re-normalize embedding
        new_emb = (base_emb + cand_emb) / 2.0
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            merged_embs_dict[i] = (new_emb / norm).astype(np.float32)

        absorbed_indices.add(j)
        merge_logs.append(p)

    final_tracks = [merged_tracks_dict[i] for i in range(len(tracks)) if i not in absorbed_indices]
    final_embs = [merged_embs_dict[i] for i in range(len(tracks)) if i not in absorbed_indices]

    return final_tracks, final_embs, merge_logs
