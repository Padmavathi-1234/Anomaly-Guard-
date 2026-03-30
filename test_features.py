import time
import subprocess
import requests
import json

server = subprocess.Popen(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"])
time.sleep(5)

try:
    BASE = 'http://localhost:7860'

    print("--- Test: Curriculum ---")
    # Doing some resets to test curriculum updating. 
    # Because baseline test didn't yield much reward, difficulty might drop.
    for _ in range(4):
        requests.post(f"{BASE}/reset", params={"task_id": 1})
        
    curriculum = requests.get(f"{BASE}/curriculum/status").json()
    print("Curriculum Status:", json.dumps(curriculum, indent=2))
    
    print("\n--- Test: Spread ---")
    spread = requests.get(f"{BASE}/metrics/spread").json()
    print("Spread Metrics:", json.dumps(spread, indent=2))
    
    print("\n--- Test: Dependencies ---")
    host = "web-server-01"
    deps = requests.get(f"{BASE}/host/{host}/dependencies").json()
    print(f"Dependencies for '{host}':", json.dumps(deps, indent=2))

    # Also check /compliance/audit to ensure nothing broke
    requests.post(f"{BASE}/reset", params={"task_id": 1})
    audit = requests.get(f"{BASE}/compliance/audit").json()
    print("\nCompliance Check Passed:", audit.get("compliant", False))
        
except Exception as e:
    print(f"Test failed: {e}")
finally:
    server.terminate()
    server.wait()
