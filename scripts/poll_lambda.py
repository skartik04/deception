#!/usr/bin/env python3
import os
import time
import json
import requests

API_KEY = os.environ["LAMBDA_API_KEY"]
INTERVAL = 60  # seconds


def check_availability() -> str | None:
    resp = requests.get(
        "https://cloud.lambdalabs.com/api/v1/instance-types",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    for name, info in data["data"].items():
        if info["regions_with_capacity_available"]:
            return name
    return None


def main() -> None:
    print("Polling Lambda Cloud for GPU availability every 60s...", flush=True)
    while True:
        try:
            available = check_availability()
            if available:
                print(f"AVAILABLE: {available}", flush=True)
                return
            print("No availability yet.", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
