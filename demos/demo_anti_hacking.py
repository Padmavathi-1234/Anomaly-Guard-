import requests
import time
import sys

def main():
    print("Starting Anti-Hacking Demo...")
    try:
        # call the endpoint
        resp = requests.post("http://localhost:7860/demo/anti-hacking")
        print("Demo Endpoint Response:", resp.json())
        
        resp = requests.get("http://localhost:7860/anti-hacking/report")
        print("Anti-Hacking Report:", resp.json())
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running on port 7860?")
        sys.exit(1)

if __name__ == "__main__":
    main()
