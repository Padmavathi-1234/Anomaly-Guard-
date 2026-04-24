import requests
import time
import sys

def main():
    print("Starting Realistic Scenarios Demo...")
    try:
        # call the realistic scenario endpoint
        resp = requests.post("http://localhost:7860/demo/realistic-scenario")
        print("Demo Endpoint Response:", resp.json())
        
        # Check live threat intel
        resp = requests.get("http://localhost:7860/threat-intel/live")
        print("Live Threat Intel:", resp.json())
        
        # Check ROI
        resp = requests.get("http://localhost:7860/business-impact/roi")
        print("Business ROI:", resp.json())
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running on port 7860?")
        sys.exit(1)

if __name__ == "__main__":
    main()
