import os
import re
import json
import subprocess
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class VideoMetadata:
    path: str
    camera: str
    start_time: str            # ISO-8601 format
    start_source: str          # 'manifest' | 'filename' | 'ffprobe' | 'manual' | 'mtime_fallback'
    fps: float
    width: int
    height: int
    duration_s: float
    rotation: int = 0


# Timestamp filename regex patterns
FILENAME_PATTERNS = [
    # YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS (e.g., 20261008_143000)
    re.compile(r"(?P<cam>[a-zA-Z0-9_\-]+?)_?(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})[_\-T](?P<hour>\d{2})(?P<minute>\d{2})(?P<sec>\d{2})"),
    # YYYY-MM-DD_HH-MM-SS
    re.compile(r"(?P<cam>[a-zA-Z0-9_\-]+?)_?(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[_\-T](?P<hour>\d{2})[:-](?P<minute>\d{2})[:-](?P<sec>\d{2})"),
    # Standalone YYYYMMDD_HHMMSS without camera prefix
    re.compile(r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})[_\-T](?P<hour>\d{2})(?P<minute>\d{2})(?P<sec>\d{2})")
]


def parse_filename_time(filename: str) -> Optional[Tuple[str, Optional[str]]]:
    """Extract (iso_timestamp, camera_name) from filename if pattern matches."""
    stem = Path(filename).stem
    for pat in FILENAME_PATTERNS:
        m = pat.search(stem)
        if m:
            gd = m.groupdict()
            try:
                dt = datetime.datetime(
                    int(gd["year"]), int(gd["month"]), int(gd["day"]),
                    int(gd["hour"]), int(gd["minute"]), int(gd["sec"])
                )
                cam = gd.get("cam", "").strip("_ -") or None
                return dt.isoformat(), cam
            except ValueError:
                continue
    return None


def run_ffprobe(video_path: Path | str, ffprobe_exe: str = "ffprobe") -> Dict[str, Any]:
    """Run ffprobe to inspect video stream tags, side data, and container format."""
    cmd = [
        ffprobe_exe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration:stream_tags:stream_side_data:format=duration,tags",
        "-of", "json",
        str(video_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)
    except Exception:
        return {}


def parse_ffprobe_rotation(probe_data: Dict[str, Any]) -> int:
    """Extract rotation angle (0, 90, 180, 270) from stream tags or side data."""
    streams = probe_data.get("streams", [])
    if not streams:
        return 0
    s = streams[0]
    
    # Check side data
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            try:
                return int(sd["rotation"]) % 360
            except (ValueError, TypeError):
                pass
                
    # Check stream tags
    tags = s.get("tags", {})
    if "rotate" in tags:
        try:
            return int(tags["rotate"]) % 360
        except (ValueError, TypeError):
            pass
            
    return 0


def parse_ffprobe_metrics(probe_data: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """Extract (width, height, fps, duration_s) from ffprobe output."""
    streams = probe_data.get("streams", [])
    fmt = probe_data.get("format", {})
    width, height, fps, duration_s = 0, 0, 30.0, 0.0
    
    if streams:
        s = streams[0]
        width = int(s.get("width", 0))
        height = int(s.get("height", 0))
        
        # Parse r_frame_rate e.g. "30/1" or "30000/1001"
        r_rate = s.get("r_frame_rate", "30/1")
        if "/" in r_rate:
            num, den = r_rate.split("/")
            fps = float(num) / max(float(den), 1.0)
        else:
            fps = float(r_rate) if r_rate else 30.0
            
        dur = s.get("duration") or fmt.get("duration")
        if dur:
            duration_s = float(dur)
            
    return width, height, fps, duration_s


def resolve_video_metadata(
    video_path: Path | str,
    manifest_path: Optional[Path | str] = None,
    manual_start_time: Optional[str] = None,
    manual_camera: Optional[str] = None,
    ffprobe_exe: Optional[str] = None
) -> VideoMetadata:
    """
    Resolve video start time and camera using 4-stage fallback chain:
    1. Explicit manual override (if supplied) -> 'manual'
    2. manifest.json lookup -> 'manifest'
    3. filename pattern parsing -> 'filename'
    4. ffprobe creation_time tag -> 'ffprobe'
    5. fallback to file mtime -> 'mtime_fallback'
    """
    v_path = Path(video_path)
    stem = v_path.stem
    filename = v_path.name
    
    # Locate ffprobe executable
    if ffprobe_exe is None:
        ffprobe_exe = os.environ.get("FFPROBE_BIN", "ffprobe")
        if not subprocess.run(f"where {ffprobe_exe}", shell=True, capture_output=True).returncode == 0:
            # Common WinGet installation fallback
            winget_path = Path(r"C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe")
            if winget_path.exists():
                ffprobe_exe = str(winget_path)
                
    probe_data = run_ffprobe(v_path, ffprobe_exe)
    width, height, fps, duration_s = parse_ffprobe_metrics(probe_data)
    rotation = parse_ffprobe_rotation(probe_data)
    
    # Stage 1: Manual override
    if manual_start_time:
        cam = manual_camera or stem
        return VideoMetadata(
            path=str(v_path),
            camera=cam,
            start_time=manual_start_time,
            start_source="manual",
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            rotation=rotation
        )

    # Stage 2: Manifest lookup
    if manifest_path is None:
        cand_manifest = v_path.parent / "manifest.json"
        if cand_manifest.exists():
            manifest_path = cand_manifest

    if manifest_path and Path(manifest_path).exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            # Lookup by filename or relative path
            entry = manifest_data.get(filename) or manifest_data.get(str(v_path))
            if entry and "start_time" in entry:
                cam = entry.get("camera") or stem
                return VideoMetadata(
                    path=str(v_path),
                    camera=cam,
                    start_time=entry["start_time"],
                    start_source=entry.get("start_source", "manifest"),
                    fps=fps,
                    width=width,
                    height=height,
                    duration_s=duration_s,
                    rotation=rotation
                )
        except Exception:
            pass

    # Stage 3: Filename pattern matching
    fn_parsed = parse_filename_time(filename)
    if fn_parsed:
        iso_time, parsed_cam = fn_parsed
        cam = manual_camera or parsed_cam or stem
        return VideoMetadata(
            path=str(v_path),
            camera=cam,
            start_time=iso_time,
            start_source="filename",
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            rotation=rotation
        )

    # Stage 4: Container metadata (creation_time in ffprobe tags)
    tags = {}
    if "streams" in probe_data and probe_data["streams"]:
        tags.update(probe_data["streams"][0].get("tags", {}))
    tags.update(probe_data.get("format", {}).get("tags", {}))
    
    creation_time = tags.get("creation_time")
    if creation_time:
        try:
            # Parse ISO or standard UTC creation time
            dt = datetime.datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
            cam = manual_camera or stem
            return VideoMetadata(
                path=str(v_path),
                camera=cam,
                start_time=dt.isoformat(),
                start_source="ffprobe",
                fps=fps,
                width=width,
                height=height,
                duration_s=duration_s,
                rotation=rotation
            )
        except Exception:
            pass

    # Stage 5: File modification time fallback
    try:
        mtime = v_path.stat().st_mtime
        dt = datetime.datetime.fromtimestamp(mtime)
        cam = manual_camera or stem
        return VideoMetadata(
            path=str(v_path),
            camera=cam,
            start_time=dt.isoformat(),
            start_source="mtime_fallback",
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            rotation=rotation
        )
    except Exception:
        # Ultimate fallback
        dt = datetime.datetime.now()
        cam = manual_camera or stem
        return VideoMetadata(
            path=str(v_path),
            camera=cam,
            start_time=dt.isoformat(),
            start_source="mtime_fallback",
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            rotation=rotation
        )
