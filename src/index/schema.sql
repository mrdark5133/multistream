-- SQLite Schema for MULTIStream (WAL mode)
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    path TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    start_time TEXT NOT NULL,       -- ISO-8601 string
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    start_source TEXT NOT NULL,     -- manifest, filename, ffprobe, manual
    rotation INTEGER DEFAULT 0,     -- 0, 90, 180, 270 degrees
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    video TEXT NOT NULL,
    camera TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    t_start TEXT NOT NULL,          -- Absolute ISO-8601
    t_end TEXT NOT NULL,            -- Absolute ISO-8601
    t_best TEXT NOT NULL,           -- Absolute ISO-8601 of best frame
    offset_start REAL NOT NULL,     -- Seconds into video
    offset_end REAL NOT NULL,
    bbox_px TEXT NOT NULL,          -- JSON [x1, y1, x2, y2]
    bbox_norm TEXT NOT NULL,        -- JSON [x1, y1, x2, y2]
    snapshot TEXT NOT NULL,         -- Relative path to JPEG
    emb BLOB NOT NULL,              -- Float32 binary bytes
    FOREIGN KEY(video) REFERENCES videos(path)
);

CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    video TEXT NOT NULL,
    camera TEXT NOT NULL,
    t_abs TEXT NOT NULL,
    offset_s REAL NOT NULL,
    snapshot TEXT NOT NULL,
    emb BLOB NOT NULL,
    FOREIGN KEY(video) REFERENCES videos(path)
);

CREATE TABLE IF NOT EXISTS aliases (
    name TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    polygon_norm TEXT,              -- JSON list of [x, y] normalized vertices or NULL
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracks_lookup ON tracks(camera, t_start, t_end);
CREATE INDEX IF NOT EXISTS idx_frames_lookup ON frames(camera, t_abs);
