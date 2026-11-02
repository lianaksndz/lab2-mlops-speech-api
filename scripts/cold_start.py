import subprocess
import time
import urllib.request
import json
import sys


def measure(container, url):
    start = time.perf_counter()

    subprocess.run(
        ["docker", "start", container],
        check=True,
        capture_output=True,
        text=True
    )

    while True:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                result = json.loads(response.read().decode())
                if response.status == 200 and result.get("status") == "ok":
                    break
        except Exception:
            time.sleep(0.1)

    elapsed = (time.perf_counter() - start) * 1000
    print(f"Container: {container}")
    print(f"Cold Start: {elapsed:.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/cold_start.py CONTAINER URL")
        sys.exit(1)

    measure(sys.argv[1], sys.argv[2])
