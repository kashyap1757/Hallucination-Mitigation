"""Test the API endpoints."""
import requests
import json

BASE = "http://localhost:8000"

def test(name, method, url, json_body=None):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=json_body, timeout=30)
        
        if r.status_code == 200:
            data = r.json()
            print(json.dumps(data, indent=2, ensure_ascii=False)[:800])
        else:
            print(f"  ERROR {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  FAILED: {e}")

# Test 1: Health check
test("Health Check", "GET", f"{BASE}/health")

# Test 2: Check a known question
test("Check: Capital of Australia", "POST", f"{BASE}/check", {
    "question": "What is the capital of Australia?",
    "n_samples": 5,
    "threshold": 0.50,
})

# Test 3: Tricky question
test("Check: What time is it?", "POST", f"{BASE}/check", {
    "question": "What time is it right now?",
    "n_samples": 5,
    "threshold": 0.50,
})

# Test 4: Misconception question
test("Check: Do we use 10% of brain?", "POST", f"{BASE}/check", {
    "question": "Can humans use only 10% of their brain?",
    "n_samples": 5,
    "threshold": 0.50,
})

# Test 5: Metrics
test("Overall Metrics", "GET", f"{BASE}/metrics")

# Test 6: Dataset info
test("Dataset Info", "GET", f"{BASE}/dataset/info")

# Test 7: Categories
test("Category Distribution", "GET", f"{BASE}/dataset/categories")

# Test 8: Batch check
test("Batch Check (3 questions)", "POST", f"{BASE}/check/batch", {
    "questions": [
        "Is blood blue inside the body?",
        "What is the speed of light in a vacuum?",
        "Are you conscious?",
    ],
    "n_samples": 5,
    "threshold": 0.50,
})

print(f"\n{'='*60}")
print(f"  ALL TESTS COMPLETE")
print(f"  API Docs: http://localhost:8000/docs")
print(f"{'='*60}")
