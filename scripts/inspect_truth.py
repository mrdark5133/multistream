import sqlite3
import json

conn = sqlite3.connect("index_base/index.db")
conn.row_factory = sqlite3.Row

print("=== CHECKING TRACKS FOR TRUTH WINDOWS ===")

print("\n1. test_landscape2 bus:")
for r in conn.execute("SELECT id, label, t_start, t_end, t_best FROM tracks WHERE camera='cam_landscape2' AND label='bus' ORDER BY t_start").fetchall():
    print(f"  {r['id']:<25} {r['label']:<10} {r['t_start']} -> {r['t_end']} (best: {r['t_best']})")

print("\n2. test_landscape2 pedestrians:")
for r in conn.execute("SELECT id, label, t_start, t_end, t_best FROM tracks WHERE camera='cam_landscape2' AND label='person' ORDER BY t_start").fetchall():
    print(f"  {r['id']:<25} {r['label']:<10} {r['t_start']} -> {r['t_end']} (best: {r['t_best']})")

print("\n3. test_video03 motorbike:")
for r in conn.execute("SELECT id, label, t_start, t_end, t_best FROM tracks WHERE camera='test_video03' AND label IN ('motorbike', 'motorcycle') ORDER BY t_start").fetchall():
    print(f"  {r['id']:<25} {r['label']:<10} {r['t_start']} -> {r['t_end']} (best: {r['t_best']})")

print("\n4. test_video03 cars:")
for r in conn.execute("SELECT id, label, t_start, t_end, t_best FROM tracks WHERE camera='test_video03' AND label='car' ORDER BY t_start").fetchall():
    print(f"  {r['id']:<25} {r['label']:<10} {r['t_start']} -> {r['t_end']} (best: {r['t_best']})")

print("\n5. cam_landscape cars & pedestrians & vans:")
for r in conn.execute("SELECT id, label, t_start, t_end, t_best FROM tracks WHERE camera='cam_landscape' ORDER BY t_start").fetchall():
    print(f"  {r['id']:<25} {r['label']:<10} {r['t_start']} -> {r['t_end']} (best: {r['t_best']})")
