
import os
import sys
from pathlib import Path

import requests

API_URL = os.getenv("API_URL", "http://localhost:8005/predict")
SAMPLES_DIR = Path("samples")

tests = {
    "test_yes.wav": "yes",
}

print("=== MODEL VERIFICATION ===")

for filename, expected in tests.items():
    filepath = SAMPLES_DIR / filename

    if not filepath.exists():
        print(f"FAIL: {filename} not found")
        sys.exit(1)

    with open(filepath, "rb") as audio:
        response = requests.post(
            API_URL,
            files={"file": (filename, audio, "audio/wav")},
            timeout=30,
        )

    if response.status_code != 200:
        print(f"FAIL: API returned {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    prediction = result["prediction"]
    confidence = result["confidence"]

    print(f"File: {filename}")
    print(f"Expected: {expected}")
    print(f"Predicted: {prediction}")
    print(f"Confidence: {confidence:.4f}")

    if prediction == expected:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

print("=== ALL TESTS PASSED ===")
