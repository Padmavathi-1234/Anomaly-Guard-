"""
OpenEnv deployment entry point.

This file is required by OpenEnv for multi-mode deployment.
It provides a standardized interface to the FastAPI application.
"""

from app.main import app, start_server

# Export app for OpenEnv deployment
__all__ = ["app"]

# Version info
__version__ = "1.0.0"

def main():
    start_server()

if __name__ == "__main__":
    main()
