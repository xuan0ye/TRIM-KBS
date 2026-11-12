"""
Evaluate one or more E2E checkpoints under fixed modality masks.

Designed for remote runs after full-train E2E checkpoints are produced:
  - p=0.3 missing audio/visual robustness across checkpoints and random seeds
  - key modality combinations (Text-only, Text+Audio, Text+Visual, Full)

This script only evaluates; it does not train.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from run_e2e_msa import (  # noqa: E402
    BERT_DEFAULT,
    DEFAULT_DATASET_REL,
    E2E_ABLATION_PRESETS,
    E2EMSAModel,
    MSARawDataset,
    build_binary_dataset,
)


KEY_COMBOS = {
    "text_only": (1, 0, 0),
    "text_audio": (1, 1, 0),
    "text_visual": (1, 0, 1),
    "full": (1, 1, 1),
}

ALL_COMBOS = {
    "text_only": (1, 0, 0),
    "audio_only": (0, 1, 0),
    "visual_only": (0, 0, 1),
    "audio_visual": (0, 1, 1),
    "text_audio": (1, 1, 0),
    "text_visual": (1, 0, 1),
    "full": (1, 1, 1),
}


def macro_f1(preds, labels, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum((p == c and y == c) for p, y in zip(preds, labels))
        fp = sum((p == c and y != c) for p, y in zip(preds, labels))
        fn = sum((p != c and y == c) for p, y in zip(preds, labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-8, precision + recall))
    return float(np.mean(f1s))


def resolve_pkl(dataset: str, data_root: str | None) -> Path:
    roots = []
    if data_root:
        roots.append(Path(data_root))
    if os.getenv("MSA_DATA_ROOT"):
        roots.append(Path(os.environ["MSA_DATA_ROOT"]))
    roots.append(Path.cwd() / "MSA Datasets")
    rel = DEFAULT_DATASET_REL[dataset]
    for root in roots:
        p = root / rel
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Cannot find {dataset} pkl under: {roots}")


def parse_ckpts(raw: str):
    paths = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            paths.append(Path(item).expanduser().resolve())
    if not paths:
        raise ValueError("--ckpts is empty")
    return paths


def forward_with_mask(model, tb, ta, tv, mask):
    bsz = tb.size(0)
    mt = mask[:, 0].to(tb.device)
    ma = mask[:, 1].to(tb.device)
    mv = mask[:, 2].to(tb.device)

    bert_out = model.bert(
        input_ids=tb[:, 0],
        attention_mask=tb[:, 1],
        token_type_ids=tb[:, 2],
        return_dict=True,
    )
    ht = model.proj_t(bert_out.last_hidden_state[:, 0, :])
    ha = model.proj_a(ta)
    hv = model.proj_v(tv)

    if hasattr(model, "use_audio") and not model.use_audio:
        ma.zero_()
    if hasattr(model, "use_visual") and not model.use_visual:
        mv.zero_()
    if getattr(model, "use_missing_repr", True):
        mt_fill = model.missing_t.expand(bsz, -1)
        ma_fill = model.missing_a.expand(bsz, -1)
        mv_fill = model.missing_v.expand(bsz, -1)
    else:
        mt_fill = torch.zeros_like(ht)
        ma_fill = torch.zeros_like(ha)
        mv_fill = torch.zeros_like(hv)
    ht = torch.where(mt.unsqueeze(-1) > 0.5, ht, mt_fill)
    ha = torch.where(ma.unsqueeze(-1) > 0.5, ha, ma_fill)
    hv = torch.where(mv.unsqueeze(-1) > 0.5, hv, mv_fill)

    if model.use_uni_ctx:
        ht_s = model.uni_t(ht.unsqueeze(1))
        ha_s = model.uni_a(ha.unsqueeze(1))
        hv_s = model.uni_v(hv.unsqueeze(1))
    else:
        ht_s, ha_s, hv_s = ht.unsqueeze(1), ha.unsqueeze(1), hv.unsqueeze(1)

    if model.use_gcm:
        ht2 = model.gcm_ta(ht_s, ha_s)
        ha2 = model.gcm_at(ha_s, ht_s)
        hv2 = model.gcm_tv(hv_s, ht_s)
    else:
        ht2, ha2, hv2 = ht_s, ha_s, hv_s

    if model.use_mm_ctx:
        if getattr(model, "use_summary_token", True):
            summ = model.mm_summary.unsqueeze(0).expand(bsz, -1, -1)
            seq = torch.cat([summ, ht2, ha2, hv2], dim=1)
            seq = model.mm_ctx(seq)
            fused = seq[:, : model.n_summary, :].reshape(bsz, -1)
        else:
            seq = torch.cat([ht2, ha2, hv2], dim=1)
            seq = model.mm_ctx(seq)
            fused = seq.reshape(bsz, -1)
    else:
        fused = torch.cat([ht2[:, 0], ha2[:, 0], hv2[:, 0]], dim=-1)
    return model.classifier(fused)


def evaluate(model, loader, n_classes, mask_builder):
    device = next(model.parameters()).device
    model.eval()
    preds, labels_all = [], []
    with torch.inference_mode():
        for tb, ta, tv, labels in loader:
            tb = tb.to(device)
            ta = ta.to(device)
            tv = tv.to(device)
            mask = mask_builder(tb.size(0)).to(device)
            logits = forward_with_mask(model, tb, ta, tv, mask)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_all.extend(labels.tolist())
    acc = sum(p == y for p, y in zip(preds, labels_all)) / len(labels_all)
    return float(acc), macro_f1(preds, labels_all, n_classes)


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate E2E checkpoints with modality masks")
    parser.add_argument("--dataset", choices=["MOSI", "MOSEI", "SIMS"], required=True)
    parser.add_argument("--ckpts", required=True, help="Comma-separated checkpoint paths")
    parser.add_argument("--data_root", default=os.getenv("MSA_DATA_ROOT"))
    parser.add_argument("--bert_model", default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--ablation", choices=sorted(E2E_ABLATION_PRESETS), default="full")
    parser.add_argument("--seeds", default="13,21,42", help="Seeds for random missing p evaluation")
    parser.add_argument("--missing_p", type=float, default=0.3)
    parser.add_argument(
        "--random_modalities",
        choices=["av", "tav"],
        default="av",
        help="Random missing evaluation: av drops audio/visual only; tav drops text/audio/visual.",
    )
    parser.add_argument("--eval_missing", action="store_true")
    parser.add_argument("--eval_combos", action="store_true")
    parser.add_argument("--combo_set", choices=["key", "all"], default="key")
    parser.add_argument("--include_empty_combo", action="store_true", help="Also evaluate T0A0V0 all-missing fixed mask.")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    if not args.eval_missing and not args.eval_combos:
        args.eval_missing = True
        args.eval_combos = True

    ckpts = parse_ckpts(args.ckpts)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    pkl_path = resolve_pkl(args.dataset, args.data_root)
    raw = pickle.load(open(pkl_path, "rb"))
    test_ds = build_binary_dataset(MSARawDataset(raw["test"]))
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    n_classes = 2
    bert_model = args.bert_model or BERT_DEFAULT[args.dataset]
    out_dir = Path(args.out_dir or f"results_e2e_mask_eval_{args.dataset.lower()}").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    print(f"dataset={args.dataset} test={len(test_ds)} a_dim={test_ds.a_dim} v_dim={test_ds.v_dim}")
    print(f"bert={bert_model} device={device}")

    for ckpt in ckpts:
        model = E2EMSAModel(
            bert_model,
            a_dim=test_ds.a_dim,
            v_dim=test_ds.v_dim,
            d_model=args.d_model,
            n_classes=n_classes,
            **E2E_ABLATION_PRESETS[args.ablation],
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        ckpt_name = ckpt.parent.name
        print(f"\nloaded {ckpt}")

        if args.eval_missing:
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                def missing_mask_builder(bsz, p=args.missing_p):
                    m = torch.ones(bsz, 3, dtype=torch.float32)
                    if args.random_modalities == "tav":
                        m[:, 0][torch.rand(bsz) < p] = 0.0
                    m[:, 1][torch.rand(bsz) < p] = 0.0
                    m[:, 2][torch.rand(bsz) < p] = 0.0
                    return m

                acc, f1 = evaluate(model, loader, n_classes, missing_mask_builder)
                rows.append({
                    "checkpoint": ckpt_name,
                    "ablation": args.ablation,
                    "mode": f"missing_{args.random_modalities}_p={args.missing_p}",
                    "seed": seed,
                    "acc": acc,
                    "f1": f1,
                    "ckpt_path": str(ckpt),
                })
                print(f"  missing seed={seed} acc={acc:.4f} f1={f1:.4f}", flush=True)

        if args.eval_combos:
            combos = ALL_COMBOS if args.combo_set == "all" else KEY_COMBOS
            if args.include_empty_combo:
                combos = {"all_missing": (0, 0, 0), **combos}
            for name, mask_tuple in combos.items():
                def combo_mask_builder(bsz, mask_tuple=mask_tuple):
                    return torch.tensor([mask_tuple], dtype=torch.float32).repeat(bsz, 1)

                acc, f1 = evaluate(model, loader, n_classes, combo_mask_builder)
                rows.append({
                    "checkpoint": ckpt_name,
                    "ablation": args.ablation,
                    "mode": name,
                    "seed": "",
                    "acc": acc,
                    "f1": f1,
                    "ckpt_path": str(ckpt),
                })
                print(f"  combo {name:<12} acc={acc:.4f} f1={f1:.4f}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    detail_csv = out_dir / f"{args.dataset}_mask_eval_detail_{ts}.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "ablation", "mode", "seed", "acc", "f1", "ckpt_path"])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for mode in sorted({r["mode"] for r in rows}):
        vals = [r for r in rows if r["mode"] == mode]
        acc_mean, acc_std = mean_std([r["acc"] for r in vals])
        f1_mean, f1_std = mean_std([r["f1"] for r in vals])
        summary.append({
            "dataset": args.dataset,
            "ablation": args.ablation,
            "mode": mode,
            "n": len(vals),
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
        })

    summary_csv = out_dir / f"{args.dataset}_mask_eval_summary_{ts}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "ablation", "mode", "n", "acc_mean", "acc_std", "f1_mean", "f1_std"],
        )
        writer.writeheader()
        writer.writerows(summary)

    report = out_dir / f"{args.dataset}_mask_eval_report_{ts}.txt"
    lines = [
        "=" * 88,
        f"  {args.dataset} E2E mask evaluation",
        "=" * 88,
        f"checkpoints={len(ckpts)}",
        f"seeds={seeds}",
        f"missing_p={args.missing_p}",
        "",
        "mode                 n    Acc(mean+/-std)      F1(mean+/-std)",
        "-" * 88,
    ]
    for r in summary:
        lines.append(
            f"{r['mode']:<20} {r['n']:<4d} "
            f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}    "
            f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f}"
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved: {detail_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {report}")


if __name__ == "__main__":
    main()
