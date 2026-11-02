import sys
import time
import json
import urllib.request
import statistics
import uuid
from pathlib import Path


def percentile(values, percent):
    values = sorted(values)
    index = int((len(values) - 1) * percent)
    return values[index]


def benchmark(url, audio_path, runs=20):
    audio = Path(audio_path).read_bytes()

    client_times = []
    server_times = []

    for i in range(runs):
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()

        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            },
            method="POST"
        )

        start = time.perf_counter()

        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode())

        elapsed = (time.perf_counter() - start) * 1000

        client_times.append(elapsed)
        server_times.append(result["latency_ms"])

        print(
            f"Запит {i + 1}/{runs}: "
            f"client={elapsed:.2f} ms, "
            f"server={result['latency_ms']:.2f} ms"
        )

    print("\nРЕЗУЛЬТАТИ BENCHMARK")
    print(f"Кількість запитів: {runs}")
    print(f"Client mean: {statistics.mean(client_times):.2f} ms")
    print(f"Client p95: {percentile(client_times, 0.95):.2f} ms")
    print(f"Server mean: {statistics.mean(server_times):.2f} ms")
    print(f"Server p95: {percentile(server_times, 0.95):.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Використання: python scripts/benchmark.py URL AUDIO.wav")
        sys.exit(1)

    benchmark(sys.argv[1], sys.argv[2])
