
import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

CLASSES = ["yes", "no", "up", "down"]


def download_dataset(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)

    archive = data_dir.parent / "speech_commands.tar.gz"

    if not archive.exists():
        print("Downloading Speech Commands dataset...")
        urllib.request.urlretrieve(URL, archive)

    print("Extracting selected classes...")

    with tarfile.open(archive, "r:gz") as tar:
        all_members = tar.getmembers()

        for class_name in CLASSES:
            members = [
                member
                for member in all_members
                if member.name.removeprefix("./").startswith(f"{class_name}/")
                and member.name.endswith(".wav")
            ]

            for member in members:
                tar.extract(member, path=data_dir)

    print("Dataset downloaded successfully.")

    for class_name in CLASSES:
        count = len(list((data_dir / class_name).glob("*.wav")))
        print(f"{class_name}: {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
    )
    args = parser.parse_args()

    download_dataset(args.data_dir)
