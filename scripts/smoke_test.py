"""
scripts/smoke_test.py
---------------------
Smoke test verification script for deployed or local HireFlow service.
Verifies GET /health, POST /run, POST /feedback, and POST /chat.
"""

import sys

import requests


def run_smoke_test(base_url: str = "http://localhost:8000"):
    print(f"Running HireFlow smoke tests against {base_url}...")

    # 1. /health
    print("Testing GET /health...", end=" ")
    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    health_data = r.json()
    assert health_data["status"] == "ok"
    print(f"PASSED (version {health_data.get('version')}, db: {health_data.get('database')})")

    # 2. /run
    print("Testing POST /run...", end=" ")
    payload = {
        "job_description": "Senior Python Engineer with 5+ years experience in FastAPI, Docker, and distributed systems. Remote.",
    }
    r = requests.post(f"{base_url}/run", json=payload, timeout=30)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    run_data = r.json()
    run_id = run_data["run_id"]
    assert len(run_data["ranked_candidates"]) > 0
    print(f"PASSED (run_id: {run_id}, ranked: {len(run_data['ranked_candidates'])})")

    # 3. /feedback
    print("Testing POST /feedback...", end=" ")
    fb_payload = {
        "candidate_email": run_data["ranked_candidates"][0]["email"],
        "job_title": "Senior Python Engineer",
        "action": "shortlisted",
        "run_id": run_id,
    }
    r = requests.post(f"{base_url}/feedback", json=fb_payload, timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    print("PASSED")

    # 4. /chat
    print("Testing POST /chat...", end=" ")
    chat_payload = {
        "message": "Find candidates for Python Engineer",
        "session_id": "smoke-test-session",
    }
    r = requests.post(f"{base_url}/chat", json=chat_payload, timeout=15)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    chat_data = r.json()
    assert "reply" in chat_data
    print("PASSED")

    print("\nAll HireFlow smoke tests passed successfully!")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_smoke_test(url)
