"""Frozen PANNs Cnn14 clip/window tagging; no emotion labels reach the encoder."""
from pathlib import Path
import argparse
import csv
import importlib.util
import subprocess
import sys
import time
import types
import numpy as np
import hashlib
from paper_common import *

SAMPLE_RATE = 32000
WINDOW = SAMPLE_RATE * 10
STRIDE = SAMPLE_RATE * 5
CHECKPOINT_MD5 = "541141fa2ee191a88f24a3219fff024e"
CHECKPOINT_NAME = "Cnn14_mAP=0.431.pth"
WATCH_EVENTS = ("Explosion", "Gunshot, gunfire", "Fireworks", "Screaming",
                "Crying, sobbing", "Laughter", "Sigh", "Applause", "Siren",
                "Speech", "Music")


def ffmpeg_executable():
    import os
    import shutil
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_audio(path, max_seconds=600):
    # Upper bound is a refusal threshold, never silent audio truncation.
    command = [ffmpeg_executable(), "-nostdin", "-v", "error", "-i", str(path),
               "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-t", str(max_seconds+1), "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=max(120, 2*max_seconds), check=False)
    if result.returncode:
        raise RuntimeError(f"Audio decode failed: {path}: {result.stderr.decode(errors='replace')[-1000:]}")
    signal = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(signal) < SAMPLE_RATE//10 or len(signal) > max_seconds*SAMPLE_RATE:
        raise ValueError(f"Refuse empty/too-short/overlong audio: {path}, seconds={len(signal)/SAMPLE_RATE}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite waveform: {path}")
    return signal


def window_starts(length):
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length-WINDOW+1, STRIDE))
    if starts[-1] != length-WINDOW:
        starts.append(length-WINDOW)
    return starts


def load_encoder(checkpoint, device):
    import torch
    import importlib.metadata
    from model_tools import runtime
    runtime(str(device), 4)
    if importlib.metadata.version("panns-inference") != "0.1.1":
        raise RuntimeError("Expected panns-inference 0.1.1.")
    if hashlib.md5(Path(checkpoint).read_bytes()).hexdigest() != CHECKPOINT_MD5:
        raise RuntimeError("PANNs checkpoint MD5 differs from the pinned Zenodo record.")
    # Load the installed package's model submodule without __init__/config:
    # upstream config otherwise downloads into ~/panns_data as an import side effect.
    package_root = Path(importlib.util.find_spec("panns_inference").origin).parent
    package = types.ModuleType("_event_panns")
    package.__path__ = [str(package_root)]
    sys.modules["_event_panns"] = package
    spec = importlib.util.spec_from_file_location("_event_panns.models", package_root/"models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                         mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def extract_signal(model, signal, device):
    import torch
    scores, embeddings = [], []
    starts = window_starts(len(signal))
    with torch.inference_mode():
        for start in starts:
            wave = signal[start:start+WINDOW]
            # Fixed ten-second windows; trailing zeros are padding, never fabricated events.
            if len(wave) < WINDOW:
                wave = np.pad(wave, (0, WINDOW-len(wave)))
            output = model(torch.from_numpy(wave[None]).to(device))
            scores.append(output["clipwise_output"][0].float().cpu().numpy())
            embeddings.append(output["embedding"][0].float().cpu().numpy())
    return (np.max(scores, axis=0).astype(np.float32),
            np.mean(embeddings, axis=0).astype(np.float32), len(starts))

