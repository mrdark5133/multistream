import sys
import time
import json
import httpx
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_api_tests():
    print("=" * 80)
    print("MULTISTREAM AUTOMATED API TEST SUITE")
    print(f"Target: {BASE_URL}")
    print("=" * 80)

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    results = []

    def log_test(name, resp, duration_ms, details=""):
        status_ok = resp.status_code in (200, 201)
        status_str = "PASS" if status_ok else "FAIL"
        print(f"[{status_str}] {name:<35} | Status: {resp.status_code} | Latency: {duration_ms:6.1f} ms | {details}")
        results.append({
            "name": name,
            "status_code": resp.status_code,
            "latency_ms": duration_ms,
            "passed": status_ok
        })

    # Test 1: GET /
    t0 = time.perf_counter()
    r1 = client.get("/")
    d1 = (time.perf_counter() - t0) * 1000.0
    log_test("GET / (Root HTML)", r1, d1, f"Content-Type: {r1.headers.get('content-type', '')}")

    # Test 2: GET /health
    t0 = time.perf_counter()
    r2 = client.get("/health")
    d2 = (time.perf_counter() - t0) * 1000.0
    h_data = r2.json() if r2.status_code == 200 else {}
    log_test("GET /health", r2, d2, f"VRAM Used: {h_data.get('vram', {}).get('used_mib', '?')} MB")

    # Test 3: GET /cameras
    t0 = time.perf_counter()
    r3 = client.get("/cameras")
    d3 = (time.perf_counter() - t0) * 1000.0
    c_data = r3.json() if r3.status_code == 200 else []
    log_test("GET /cameras", r3, d3, f"Found {len(c_data)} cameras")

    # Test 4: GET /aliases
    t0 = time.perf_counter()
    r4 = client.get("/aliases")
    d4 = (time.perf_counter() - t0) * 1000.0
    a_data = r4.json() if r4.status_code == 200 else {}
    log_test("GET /aliases", r4, d4, f"Found {len(a_data)} aliases")

    # Test 5: POST /ask (Unknown referent -> clarify)
    unique_place = f"test courtyard {int(time.time())}"
    t0 = time.perf_counter()
    r5 = client.post("/ask", json={"query": f"a person at the {unique_place}", "session_id": "test_sess"})
    d5 = (time.perf_counter() - t0) * 1000.0
    q5_data = r5.json() if r5.status_code == 200 else {}
    status_5 = q5_data.get("status")
    log_test("POST /ask (Clarify Unknown)", r5, d5, f"Status: {status_5}, Referent: '{q5_data.get('referent')}'")
    assert status_5 == "clarify", f"Expected clarify status, got {status_5}"

    # Test 6: POST /alias (Register alias with polygon)
    t0 = time.perf_counter()
    poly = [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]]
    r6 = client.post("/alias", json={"name": unique_place, "camera": "cam_landscape", "polygon": poly})
    d6 = (time.perf_counter() - t0) * 1000.0
    log_test("POST /alias (Register Spatial)", r6, d6, f"Saved '{unique_place}' -> cam_landscape")

    # Test 7: POST /ask (Resolved alias -> success)
    t0 = time.perf_counter()
    r7 = client.post("/ask", json={"query": f"a person at the {unique_place}", "session_id": "test_sess"})
    d7 = (time.perf_counter() - t0) * 1000.0
    q7_data = r7.json() if r7.status_code == 200 else {}
    status_7 = q7_data.get("status")
    log_test("POST /ask (Resolved Query)", r7, d7, f"Status: {status_7}, Cam: {q7_data.get('parsed', {}).get('resolved_camera')}")
    assert status_7 == "success", f"Expected success status, got {status_7}"

    # Test 8: POST /ask (Standard query for real tracks)
    t0 = time.perf_counter()
    r8 = client.post("/ask", json={"query": "a red car near cam_landscape", "top_k": 3})
    d8 = (time.perf_counter() - t0) * 1000.0
    q8_data = r8.json() if r8.status_code == 200 else {}
    n_res = len(q8_data.get("results", []))
    log_test("POST /ask (Retrieve Vehicle)", r8, d8, f"Found {n_res} candidate results")

    first_trk_id = q8_data["results"][0]["id"] if n_res > 0 else "test_landscape_trk_108"

    # Test 9: GET /snapshot/{id}
    t0 = time.perf_counter()
    r9 = client.get(f"/snapshot/{first_trk_id}")
    d9 = (time.perf_counter() - t0) * 1000.0
    log_test(f"GET /snapshot/{first_trk_id}", r9, d9, f"Bytes: {len(r9.content)}")

    # Test 10: GET /clip/{id}
    t0 = time.perf_counter()
    r10 = client.get(f"/clip/{first_trk_id}")
    d10 = (time.perf_counter() - t0) * 1000.0
    log_test(f"GET /clip/{first_trk_id}", r10, d10, f"Bytes: {len(r10.content)}")

    # Test 11: POST /upload
    test_upload_file = Path("footage/test_video01.mp4")
    if test_upload_file.exists():
        t0 = time.perf_counter()
        with open(test_upload_file, "rb") as f:
            r11 = client.post(
                "/upload",
                files={"file": ("test_upload_sample.mp4", f.read()[:50000], "video/mp4")},
                data={"camera": "cam_upload_test", "start_time": "2026-10-08T18:00:00", "rotation": 0}
            )
        d11 = (time.perf_counter() - t0) * 1000.0
        log_test("POST /upload", r11, d11, f"Registered camera 'cam_upload_test'")

    print("=" * 80)
    all_passed = all(r["passed"] for r in results)
    print(f"SUMMARY: {sum(r['passed'] for r in results)} / {len(results)} API tests PASSED.")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = run_api_tests()
    sys.exit(0 if success else 1)
