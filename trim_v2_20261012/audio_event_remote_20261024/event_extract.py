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
from event_common import *

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
    from runtime_v2 import runtime
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


def feature_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=HERE/"assets")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    plan = read_json(args.out/"inventory.json")
    checkpoint = args.assets/CHECKPOINT_NAME
    with (args.assets/"class_labels_indices.csv").open(encoding="utf-8-sig", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    if len(label_rows) != 527 or [int(x["index"]) for x in label_rows] != list(range(527)):
        raise ValueError("Expected the canonical 527-class ordered AudioSet label list.")
    labels = [x["display_name"] for x in label_rows]
    signature = dict(release=RELEASE, data_sha256=plan["data_sha256"], smoke=plan["smoke"],
                     probe=plan.get("probe",False),
                     checkpoint_sha256=sha_file(checkpoint),
                     labels_sha256=sha_file(args.assets/"class_labels_indices.csv"),
                     extractor_sha256=sha_file(Path(__file__)), sample_rate=SAMPLE_RATE,
                     window_seconds=10, stride_seconds=5,
                     event_pool="maximum across windows", embedding_pool="mean across windows",
                     torch=torch.__version__, device_type=device.type, cudnn=False)
    signature_path = args.out/"extraction_signature.json"
    if signature_path.exists() and read_json(signature_path) != signature:
        raise RuntimeError("Extraction settings changed; use a new run directory.")
    dump_json(signature_path, signature)
    model = load_encoder(checkpoint, device)
    all_records, preview = [], []
    total = sum(map(len, plan["splits"].values()))
    done = 0
    began = time.monotonic()
    for split, records in plan["splits"].items():
        event_rows, embed_rows, ids, durations, window_counts = [], [], [], [], []
        for record in records:
            path = Path(plan["raw_root"])/record["relative_path"]
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise RuntimeError(f"Media changed since inventory: {path}; rerun audit.")
            digest = sha_file(path)
            cache = args.out/"clip_cache"/(feature_key(record["id"])+".npz")
            if cache.exists():
                item = load_npz(cache)
                if str(item["source_sha256"].item()) != digest:
                    raise RuntimeError(f"Cached media content changed: {path}")
                event, embed = item["events"], item["embedding"]
                seconds, windows, rms = float(item["seconds"]), int(item["windows"]), float(item["rms"])
            else:
                signal = decode_audio(path)
                event, embed, windows = extract_signal(model, signal, device)
                seconds = len(signal)/SAMPLE_RATE
                rms = float(np.sqrt(np.mean(signal.astype(np.float64)**2)))
                atomic_npz(cache, events=event, embedding=embed, seconds=np.array(seconds),
                           windows=np.array(windows), rms=np.array(rms), source_sha256=np.array(digest))
            if event.shape != (527,) or embed.shape != (2048,) or not np.isfinite(embed).all() or not np.isfinite(event).all() or not ((event >= 0) & (event <= 1)).all():
                raise ValueError(f"Invalid PANNs cache: {cache}")
            event_rows.append(event);embed_rows.append(embed);ids.append(record["id"])
            durations.append(seconds);window_counts.append(windows)
            all_records.append(dict(split=split, id=record["id"], source_sha256=digest,
                                    cache_sha256=sha_file(cache), relative_path=record["relative_path"],
                                    seconds=seconds, windows=windows, rms=rms))
            if split == "train":
                top = np.argsort(event)[-5:][::-1]
                preview.append(dict(id=record["id"], relative_path=record["relative_path"],
                                    seconds=seconds, rms=rms, top_events=" | ".join(f"{labels[i]}={event[i]:.4f}" for i in top)))
            done += 1
            if done == 1 or done % 25 == 0 or done == total:
                elapsed = time.monotonic()-began
                dump_json(args.out/"extraction_progress.json", dict(done=done,total=total,split=split,status="extracting",
                          elapsed_minutes=round(elapsed/60,2),estimated_remaining_minutes=round(elapsed/done*(total-done)/60,2)))
                print("EVENT_EXTRACT",done,"/",total,split,record["id"],flush=True)
        atomic_npz(args.out/f"events_{split}.npz", ids=np.asarray(ids), events=np.stack(event_rows),
                   embedding=np.stack(embed_rows), seconds=np.asarray(durations),
                   windows=np.asarray(window_counts), labels=np.asarray(labels))
    # Coverage is computed on train only. This is detector output, not human event truth.
    train = load_npz(args.out/"events_train.npz")
    coverage = []
    for index, name in enumerate(labels):
        values = train["events"][:, index]
        coverage.append(dict(event=name,index=index,n_train=len(values),mean_score=float(values.mean()),
                             max_score=float(values.max()),count_ge_0_3=int((values>=.3).sum()),
                             count_ge_0_5=int((values>=.5).sum()),watch_event=name in WATCH_EVENTS))
    write_csv(args.out/"event_coverage_train.csv", coverage)
    # Random, label-free listening sample; probabilities alone are not validated accuracy.
    rng = np.random.default_rng(SEED)
    sample = [preview[i] for i in rng.permutation(len(preview))[:min(200,len(preview))]]
    write_csv(args.out/"listening_sample_train.csv", sample)
    dump_json(args.out/"media_manifest.json", dict(records=all_records,signature=signature,test_evaluated=False))
    output_names = [f"events_{split}.npz" for split in plan["splits"]]
    output_names += ["media_manifest.json","event_coverage_train.csv","listening_sample_train.csv"]
    manifest = {name:sha_file(args.out/name) for name in output_names}
    dump_json(args.out/"extraction_complete.json", dict(status="complete",files=manifest,signature=signature,
              test_evaluated=False,elapsed_seconds=time.monotonic()-began,
              note="Window/clip tags, not time-localized events; scores are uncalibrated and not emotion labels."))
    dump_json(args.out/"extraction_progress.json",dict(status="complete",done=total,total=total))
    print("EVENT_EXTRACTION_COMPLETE",total,"test_evaluated",False,flush=True)
    if plan.get("probe",False):
        import shutil
        import zipfile
        review = args.out/"probe_review"
        review.mkdir(exist_ok=True)
        names = output_names+["inventory.json","audit.json","extraction_signature.json",
                              "extraction_complete.json","extraction_progress.json"]
        for name in names:
            shutil.copy2(args.out/name,review/name)
        dump_json(review/"review_manifest.json",dict(probe=True,scope="available_training_media_only",
                  test_evaluated=False,contains_emotion_performance_results=False,
                  files={name:sha_file(review/name) for name in names}))
        archive = args.out/"event_probe_20261024.zip"
        temporary = archive.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(review.iterdir()):
                zipped.write(path,"probe_review/"+path.name)
        temporary.replace(archive)
        print("EVENT_PROBE_REVIEW_COMPLETE",archive,flush=True)


if __name__ == "__main__":
    main()
