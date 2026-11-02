import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from model import (
    SmallCNN,
    CLASSES,
    SAMPLE_RATE,
    NUM_SAMPLES,
    create_mel_transform
)


class SpeechCommandsDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.mel_transform = create_mel_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        audio, sample_rate = sf.read(
            str(file_path),
            dtype="float32",
            always_2d=True
        )

        # soundfile повертає форму (кількість семплів, канали).
        # Перетворюємо її на форму (канали, кількість семплів).
        waveform = torch.from_numpy(audio.T.copy())

        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Перетворення стерео в моно.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Доповнення або обрізання до однієї секунди.
        if waveform.shape[1] < NUM_SAMPLES:
            padding = NUM_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(
                waveform,
                (0, padding)
            )
        else:
            waveform = waveform[:, :NUM_SAMPLES]

        spectrogram = self.mel_transform(waveform)
        spectrogram = torch.log(spectrogram + 1e-6)

        return spectrogram, label


def collect_audio_files(data_dir):
    samples = []

    for class_index, class_name in enumerate(CLASSES):
        class_dir = Path(data_dir) / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Не знайдено папку класу: {class_dir}"
            )

        files = list(class_dir.glob("*.wav"))

        for file_path in files:
            samples.append((file_path, class_index))

        print(f"{class_name}: знайдено {len(files)} файлів")

    if not samples:
        raise RuntimeError("Аудіофайли не знайдені")

    return samples


def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Пристрій для навчання: {device}")

    start_time = time.perf_counter()

    samples = collect_audio_files(args.data_dir)
    random.Random(args.seed).shuffle(samples)

    split_index = int(len(samples) * 0.8)
    train_samples = samples[:split_index]
    test_samples = samples[split_index:]

    print(f"\nУсього файлів: {len(samples)}")
    print(f"Навчальна вибірка: {len(train_samples)}")
    print(f"Тестова вибірка: {len(test_samples)}")

    train_dataset = SpeechCommandsDataset(train_samples)
    test_dataset = SpeechCommandsDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SmallCNN(num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Епоха {epoch + 1}/{args.epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    accuracy = evaluate(model, test_loader, device)

    training_time = time.perf_counter() - start_time

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": CLASSES,
            "sample_rate": SAMPLE_RATE
        },
        model_path
    )

    model_size_kb = model_path.stat().st_size / 1024

    metrics = {
        "accuracy_percent": round(accuracy * 100, 2),
        "training_time_seconds": round(training_time, 2),
        "model_size_kb": round(model_size_kb, 2),
        "epochs": args.epochs,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "device": str(device)
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    with open(
        output_dir / "labels.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(CLASSES, file, ensure_ascii=False, indent=4)

    print("\nНавчання завершено!")
    print(f"Точність: {accuracy * 100:.2f}%")
    print(f"Час навчання: {training_time:.2f} секунд")
    print(f"Розмір моделі: {model_size_kb:.2f} KB")
    print(f"Модель збережена: {model_path}")


if __name__ == "__main__":
    main()