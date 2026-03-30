import time
import subprocess
import requests
import json

# Start the server
server = subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"])
time.sleep(5) # Wait for server to start

try:
    BASE = 'http://localhost:7860'

    # Test 1 and 2 Setup
    print("--- Setup ---")
    requests.post(f"{BASE}/reset", params={"task_id": 1, "seed": 42})
    requests.post(f"{BASE}/baseline", params={"task_id": 1, "seed": 42})

    # Test 1
    print("\n--- Test 1: /compliance/audit (Snippet) ---")
    audit1 = requests.get(f"{BASE}/compliance/audit").json()
    print(json.dumps({k: v for k, v in audit1.items() if k != "compliance_checks"}, indent=2))

    # Test 2
    print("\n--- Test 2: /compliance/trail ---")
    trail = requests.get(f"{BASE}/compliance/trail").json()
    action_length = len(trail.get('actions', []))
    print(f"Trail returned. Number of actions: {action_length}")

    # Test 3
    print("\n--- Test 3: Determinism Check ---")
    requests.post(f'{BASE}/reset', params={'task_id': 1, 'seed': 42})
    requests.post(f'{BASE}/baseline', params={'task_id': 1, 'seed': 42})
    audit2 = requests.get(f'{BASE}/compliance/audit').json()

    # Check determinism
    assert audit1['risk_level'] == audit2['risk_level'], 'Risk level differs!'
    assert audit1['compliant'] == audit2['compliant'], 'Compliant status differs!'
    assert audit1['all_actions_justified'] == audit2['all_actions_justified'], 'Justification check differs!'

    print('✅ Audit report is deterministic')
    print(f'   Risk level: {audit1["risk_level"]}')
    print(f'   Compliant: {audit1["compliant"]}')
    print(f'   Checks passed: {sum(1 for c in audit1["compliance_checks"] if c["passed"])}/5')

except Exception as e:
    print(f"Test failed: {e}")
finally:
    server.terminate()
    server.wait()
