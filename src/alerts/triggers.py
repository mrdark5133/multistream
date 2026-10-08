import sqlite3
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.index.db import blob_to_emb
from src.query.reid import cosine_similarity


def ensure_alerts_schema(conn: sqlite3.Connection) -> None:
    """Ensure alert tables exist in SQLite database."""
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS alert_rules (
            id TEXT PRIMARY KEY,
            rule_name TEXT NOT NULL,
            query_text TEXT NOT NULL,
            camera_filter TEXT,
            min_score REAL DEFAULT 0.60,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            last_triggered_at TEXT
        );

        CREATE TABLE IF NOT EXISTS alert_events (
            id TEXT PRIMARY KEY,
            rule_id TEXT NOT NULL,
            track_id TEXT NOT NULL,
            camera TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL NOT NULL,
            snapshot_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES alert_rules(id),
            FOREIGN KEY(track_id) REFERENCES tracks(id)
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alert_events(rule_id, timestamp);
        """)


def create_alert_rule(
    conn: sqlite3.Connection,
    rule_name: str,
    query_text: str,
    camera_filter: Optional[str] = None,
    min_score: float = 0.05
) -> str:
    """Create a new persistent standing alert rule."""
    ensure_alerts_schema(conn)
    rule_id = f"rule_{uuid.uuid4().hex[:8]}"
    now_iso = datetime.datetime.now().isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO alert_rules (id, rule_name, query_text, camera_filter, min_score, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?);
            """,
            (rule_id, rule_name, query_text, camera_filter, min_score, now_iso)
        )
    return rule_id


def list_alert_rules(conn: sqlite3.Connection, active_only: bool = True) -> List[Dict[str, Any]]:
    """List standing alert rules."""
    ensure_alerts_schema(conn)
    sql = "SELECT * FROM alert_rules"
    if active_only:
        sql += " WHERE is_active = 1"
    sql += " ORDER BY created_at DESC;"
    cur = conn.execute(sql)
    return [dict(r) for r in cur.fetchall()]


def delete_alert_rule(conn: sqlite3.Connection, rule_id: str) -> bool:
    """Delete an alert rule and its triggered events."""
    ensure_alerts_schema(conn)
    with conn:
        conn.execute("DELETE FROM alert_events WHERE rule_id = ?;", (rule_id,))
        cur = conn.execute("DELETE FROM alert_rules WHERE id = ?;", (rule_id,))
        return cur.rowcount > 0


def evaluate_alert_rules(
    conn: sqlite3.Connection,
    embedder,
    rules: Optional[List[Dict[str, Any]]] = None,
    track_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate active standing alert rules against tracks in the database.
    If track_ids is provided, evaluates only against those tracks (incremental/online mode).
    Otherwise evaluates against all tracks in the database.
    """
    ensure_alerts_schema(conn)

    if rules is None:
        rules = list_alert_rules(conn, active_only=True)

    if not rules:
        return []

    # Fetch candidate tracks
    if track_ids:
        placeholders = ",".join("?" for _ in track_ids)
        cur = conn.execute(f"SELECT * FROM tracks WHERE id IN ({placeholders});", track_ids)
    else:
        cur = conn.execute("SELECT * FROM tracks;")
    tracks = cur.fetchall()

    triggered_events = []
    now_iso = datetime.datetime.now().isoformat()

    # Pre-embed text queries for each rule to avoid repeated model calls
    rule_embeddings = {}
    for rule in rules:
        emb = embedder.embed_text([rule["query_text"]])[0]
        rule_embeddings[rule["id"]] = emb

    for rule in rules:
        r_id = rule["id"]
        q_emb = rule_embeddings[r_id]
        min_score = rule["min_score"]
        cam_filter = rule.get("camera_filter")

        for trk in tracks:
            if cam_filter and trk["camera"] != cam_filter:
                continue

            trk_emb = blob_to_emb(trk["emb"])
            score = cosine_similarity(q_emb, trk_emb)

            if score >= min_score:
                # Check duplicate trigger
                dup_cur = conn.execute(
                    "SELECT id FROM alert_events WHERE rule_id = ? AND track_id = ?;",
                    (r_id, trk["id"])
                )
                if dup_cur.fetchone():
                    continue

                event_id = f"evt_{uuid.uuid4().hex[:8]}"
                with conn:
                    conn.execute(
                        """
                        INSERT INTO alert_events (id, rule_id, track_id, camera, timestamp, score, snapshot_path, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (event_id, r_id, trk["id"], trk["camera"], trk["t_start"], round(score, 4), trk["snapshot"], now_iso)
                    )
                    conn.execute(
                        "UPDATE alert_rules SET last_triggered_at = ? WHERE id = ?;",
                        (now_iso, r_id)
                    )

                triggered_events.append({
                    "event_id": event_id,
                    "rule_id": r_id,
                    "rule_name": rule["rule_name"],
                    "query_text": rule["query_text"],
                    "track_id": trk["id"],
                    "camera": trk["camera"],
                    "label": trk["label"],
                    "score": round(score, 4),
                    "timestamp": trk["t_start"],
                    "snapshot": trk["snapshot"]
                })

    return triggered_events


def get_alert_events(
    conn: sqlite3.Connection,
    rule_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Retrieve triggered alert events from the database."""
    ensure_alerts_schema(conn)
    if rule_id:
        cur = conn.execute(
            """
            SELECT e.*, r.rule_name, r.query_text
            FROM alert_events e
            JOIN alert_rules r ON e.rule_id = r.id
            WHERE e.rule_id = ?
            ORDER BY e.timestamp DESC
            LIMIT ?;
            """,
            (rule_id, limit)
        )
    else:
        cur = conn.execute(
            """
            SELECT e.*, r.rule_name, r.query_text
            FROM alert_events e
            JOIN alert_rules r ON e.rule_id = r.id
            ORDER BY e.timestamp DESC
            LIMIT ?;
            """,
            (limit,)
        )
    return [dict(r) for r in cur.fetchall()]
