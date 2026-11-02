import torch
import torch.nn as nn
import torchaudio


SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
N_MELS = 64

CLASSES = ["yes", "no", "up", "down"]


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def create_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    )


def prepare_audio(waveform, original_sample_rate, mel_transform):
    # Перетворення частоти дискретизації
    if original_sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Перетворення в моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Вирівнювання довжини аудіо до 1 секунди
    if waveform.shape[1] < NUM_SAMPLES:
        padding = NUM_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :NUM_SAMPLES]

    # Mel-спектрограма
    spectrogram = mel_transform(waveform)

    # Логарифмування
    spectrogram = torch.log(spectrogram + 1e-6)

    return spectrogram.unsqueeze(0)