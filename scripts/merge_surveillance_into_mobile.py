import sqlite3
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

fresh_db_path = ROOT_DIR / "index_fresh" / "index.db"
fresh_snaps_dir = ROOT_DIR / "index_fresh" / "snapshots"

mobile_db_path = ROOT_DIR / "index_mobile" / "index.db"
mobile_snaps_dir = ROOT_DIR / "index_mobile" / "snapshots"

print("=" * 80)
print("MERGING SURVEILLANCE FOOTAGE (VEHICLES & PERSONS) INTO INDEX_MOBILE")
print("=" * 80)

c_fresh = sqlite3.connect(fresh_db_path)
c_mob = sqlite3.connect(mobile_db_path)

# 1. Copy surveillance video entries
surveillance_videos = c_fresh.execute(
    "SELECT path, camera, start_time, fps, width, height, duration_s, start_source, rotation, indexed_at FROM videos WHERE path NOT LIKE '%mobile_cam%'"
).fetchall()

print(f"Found {len(surveillance_videos)} surveillance videos in index_fresh.")
with c_mob:
    for row in surveillance_videos:
        c_mob.execute("""
            INSERT OR REPLACE INTO videos (path, camera, start_time, fps, width, height, duration_s, start_source, rotation, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
        print(f"  + Added video: {row[0]} ({row[1]})")

# 2. Copy surveillance tracks (cars, persons, trucks, buses, etc.)
track_cols = ['id', 'video', 'camera', 'track_id', 'label', 'group', 't_start', 't_end', 't_best', 'offset_start', 'offset_end', 'offset_best', 'bbox_px', 'bbox_norm', 'snapshot', 'emb', 'colors', 'quality', 'hits']
cols_str = ", ".join([f'"{c}"' for c in track_cols])
placeholders = ", ".join(["?"] * len(track_cols))

surveillance_tracks = c_fresh.execute(
    f"SELECT {cols_str} FROM tracks WHERE video NOT LIKE '%mobile_cam%'"
).fetchall()

print(f"\nFound {len(surveillance_tracks)} surveillance tracks in index_fresh.")
with c_mob:
    for trk in surveillance_tracks:
        c_mob.execute(f"INSERT OR REPLACE INTO tracks ({cols_str}) VALUES ({placeholders})", trk)

print(f"  + Added {len(surveillance_tracks)} tracks into index_mobile.")

# 3. Copy surveillance frames
frame_cols = [c[1] for c in c_fresh.execute("PRAGMA table_info(frames)").fetchall()]
f_cols_str = ", ".join([f'"{c}"' for c in frame_cols])
f_placeholders = ", ".join(["?"] * len(frame_cols))

surveillance_frames = c_fresh.execute(
    f"SELECT {f_cols_str} FROM frames WHERE video NOT LIKE '%mobile_cam%'"
).fetchall()

with c_mob:
    for frm in surveillance_frames:
        c_mob.execute(f"INSERT OR REPLACE INTO frames ({f_cols_str}) VALUES ({f_placeholders})", frm)

print(f"  + Added {len(surveillance_frames)} frames into index_mobile.")

# 4. Copy snapshot directories from index_fresh/snapshots to index_mobile/snapshots
surveillance_dirs = ['test_landscape', 'test_landscape2', 'test_video01', 'test_video02', 'test_video03']
for d in surveillance_dirs:
    src_dir = fresh_snaps_dir / d
    dst_dir = mobile_snaps_dir / d
    if src_dir.exists():
        if not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)
            print(f"  + Copied snapshot folder: {d}")
        else:
            print(f"  * Snapshot folder already exists: {d}")

# 5. Verify total counts in index_mobile
total_videos = c_mob.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
total_tracks = c_mob.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
total_frames = c_mob.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

print("\n" + "=" * 80)
print(f"INDEX_MOBILE SUMMARY (ALL CAMERAS):")
print(f"Total Videos: {total_videos}")
print(f"Total Tracks: {total_tracks}")
print(f"Total Frames: {total_frames}")

print("\nTrack breakdown by label in index_mobile:")
for r in c_mob.execute("SELECT label, count(*) FROM tracks GROUP BY label ORDER BY count(*) DESC").fetchall():
    print(f"  {r[0]}: {r[1]}")
print("=" * 80)
