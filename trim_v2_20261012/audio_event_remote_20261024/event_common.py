"""Shared provenance and strict MOSEI CSV/PKL alignment; no test evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE / "_core"
sys.path.insert(0, str(CORE))
from common import sha_file, dump_json, write_csv
import mmap_dataset_pickle as mm

PATTERNS = ("text_only", "text_audio", "text_visual", "full")
RELEASE = "audio_event_20261024"
SEED = 42


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def data_file(root):
    return Path(root) / "MSA Datasets/MOSEI/Processed/unaligned_50.pkl"


def load_raw(root):
    with data_file(root).open("rb") as handle:
        return mm.load(handle)


def selected(raw, split, limit=None):
    if split not in ("train", "valid"):
        raise ValueError("This pilot permits train and valid only.")
    part = raw[split]
    labels = np.asarray(part["classification_labels"]).reshape(-1)
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Invalid original sentiment labels.")
    indices = np.flatnonzero(labels != 1)
    y = (labels[indices] == 2).astype(np.int64)
    if limit is not None:
        rng = np.random.default_rng(SEED)
        half = limit // 2
        pick = np.r_[rng.choice(np.flatnonzero(y == 0), half, replace=False),
                     rng.choice(np.flatnonzero(y == 1), limit-half, replace=False)]
        indices, y = indices[pick], y[pick]
    ids = np.asarray([str(part["id"][i]) for i in indices])
    reg = np.asarray(part["regression_labels"]).reshape(-1)[indices]
    if len(set(ids.tolist())) != len(ids) or not np.array_equal(reg > 0, y == 1):
        raise ValueError("Duplicate IDs or inconsistent regression labels.")
    return indices, ids, y


def relative_media(sample_id):
    fields = str(sample_id).split("$_$")
    if len(fields) != 2 or any(not x or x in (".", "..") or "/" in x or "\\" in x for x in fields):
        raise ValueError(f"Unsupported sample ID: {sample_id!r}")
    return Path(fields[0]) / (fields[1] + ".mp4")


def derangement(n, seed=20261024):
    if n < 2:
        raise ValueError("At least two rows required for a derangement.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    result = np.empty(n, dtype=np.int64)
    result[order] = np.roll(order, 1)
    assert np.all(result != np.arange(n))
    return result


def project_features(train, valid, width=128):
    """Fit scaling on train only, then apply a fixed label-free projection."""
    train, valid = np.asarray(train, np.float32), np.asarray(valid, np.float32)
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1]:
        raise ValueError("Feature dimensions differ.")
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise ValueError("Nonfinite event input.")
    mean = train.astype(np.float64).mean(0).astype(np.float32)
    std = np.maximum(train.astype(np.float64).std(0), 1e-4).astype(np.float32)
    matrix = np.random.default_rng(20261024).normal(
        size=(train.shape[1], width)).astype(np.float32) / np.sqrt(train.shape[1])
    transform = lambda x: (np.clip((x-mean)/std, -10, 10) @ matrix).astype(np.float32)
    return transform(train), transform(valid), dict(mean=mean, std=std, projection=matrix)


def build_inventory(root, raw_root, out, smoke=False, probe=False, sample_count=200):
    root, raw_root, out = Path(root), Path(raw_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw(root)
    label_path = root / "MSA Datasets/MOSEI/label.csv"
    with label_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {}
    for row in rows:
        key = row["video_id"] + "$_$" + row["clip_id"]
        if key in lookup:
            raise ValueError(f"Duplicate CSV ID: {key}")
        lookup[key] = row
    plans, missing, all_ids, availability = {}, [], set(), {}
    for split in (("train",) if probe else ("train", "valid")):
        indices, ids, y = selected(raw, split, 16 if smoke else None)
        if all_ids & set(ids.tolist()):
            raise ValueError("Train/validation IDs overlap.")
        all_ids.update(ids.tolist())
        records = []
        for i, sid, target in zip(indices, ids, y):
            row = lookup.get(str(sid))
            reg = float(np.asarray(raw[split]["regression_labels"]).reshape(-1)[i])
            if row is None or row["mode"] != split or abs(float(row["label"])-reg) > 1e-5:
                raise ValueError(f"CSV/PKL split or label mismatch: {split}/{sid}")
            path = raw_root / relative_media(sid)
            record = dict(id=str(sid), relative_path=relative_media(sid).as_posix(),
                          size_bytes=path.stat().st_size if path.is_file() else 0)
            records.append(record)
            if record["size_bytes"] == 0:
                missing.append(dict(split=split, id=str(sid), path=str(path)))
        available = [r for r in records if r["size_bytes"] > 0]
        availability[split] = dict(total=len(records),available=len(available),
                                   missing=len(records)-len(available))
        if probe:
            if not available:
                raise FileNotFoundError(f"No available training media under {raw_root}")
            choice = np.random.default_rng(SEED).permutation(len(available))[:sample_count]
            records = [available[i] for i in choice]
        plans[split] = records
    print("EVENT_HASH_DATA",str(data_file(root)),flush=True)
    plan = dict(release=RELEASE, dataset="MOSEI", view="native", smoke=smoke,
                probe=probe, protocol="available_train_only_probe" if probe else "complete_non_neutral_train_valid",
                availability=availability,
                data_sha256=sha_file(data_file(root)), csv_sha256=sha_file(label_path),
                raw_root=str(raw_root.resolve()), splits=plans,
                test_evaluated=False, test_media_requested=False)
    dump_json(out/"inventory.json", plan)
    if missing:
        write_csv(out/"missing_media.csv", missing)
    report = dict(status="probe_paths_verified" if probe else ("missing_media" if missing else "paths_verified"),
                  counts={k:len(v) for k,v in plans.items()}, missing=len(missing),
                  availability=availability,probe=probe,
                  audio_decoded=False, data_sha256=plan["data_sha256"],
                  next_step="extract" if not missing else "Upload missing_media.csv paths with Xftp; rerun audit.",
                  test_evaluated=False)
    dump_json(out/"audit.json", report)
    print("EVENT_PATH_AUDIT", json.dumps(report), flush=True)
    if missing and not probe:
        raise FileNotFoundError(f"{len(missing)} media files missing; see {out/'missing_media.csv'}")
    return plan


def verify_bundle():
    manifest = read_json(HERE/"bundle_manifest.json")
    for name, expected in manifest["files"].items():
        if sha_file(HERE/name) != expected:
            raise RuntimeError(f"Upload checksum mismatch: {name}")
    print("EVENT_CODE_VERIFIED", len(manifest["files"]), flush=True)


def command():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "audit"))
    parser.add_argument("--root", type=Path, default=HERE.parents[1])
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--out", type=Path, default=HERE/"runs/pilot")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--sample-count",type=int,default=200)
    args = parser.parse_args()
    if args.mode == "verify":
        verify_bundle()
    else:
        if args.sample_count < 1 or (args.probe and args.smoke):
            parser.error("sample-count must be positive; probe and smoke are separate.")
        build_inventory(args.root, args.raw_root or args.root/"MSA Datasets/MOSEI/Raw",
                        args.out, args.smoke,args.probe,args.sample_count)


if __name__ == "__main__":
    command()
