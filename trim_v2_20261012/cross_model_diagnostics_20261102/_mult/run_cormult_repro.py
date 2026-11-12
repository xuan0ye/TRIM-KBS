from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
CORMULT_ROOT = ROOT / "CorMulT-main"
sys.path.insert(0, str(CORMULT_ROOT))
sys.path.insert(0, str(CORMULT_ROOT / "modality_correlation"))

from modality_correlation.correlation_loss import TripleLoss  # noqa: E402
from modality_correlation.correlation_models import CorrelationModel  # noqa: E402
from modules.transformer import TransformerEncoder  # noqa: E402


DEFAULT_DATASET_REL = {
    "MOSI": Path("MOSI/Processed/aligned_50.pkl"),
    "MOSEI": Path("MOSEI/Processed/aligned_50.pkl"),
    "SIMS": Path("SIMS/Processed/unaligned_39.pkl"),
}


def resolve_dataset_pkl(dataset: str, data_root: str | None) -> Path:
    roots = []
    if data_root:
        roots.append(Path(data_root))
    if os.getenv("MSA_DATA_ROOT"):
        roots.append(Path(os.environ["MSA_DATA_ROOT"]))
    roots.append(ROOT / "MSA Datasets")
    rel = DEFAULT_DATASET_REL[dataset]
    for root in roots:
        p = root / rel
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Cannot find {dataset} pkl under {roots}")


class VWBSequenceDataset(Dataset):
    """Unified binary MSA dataset for CorMulT reproduction."""

    def __init__(self, split_data: dict, max_text_len: int = 50, max_audio_len: int = 100, max_vision_len: int = 100):
        ids = split_data.get("id", list(range(len(split_data["classification_labels"]))))
        texts = split_data["text"]
        audios = split_data["audio"]
        visions = split_data["vision"]
        labels = split_data.get("classification_labels", split_data.get("regression_labels"))

        self.items = []
        for i in range(len(labels)):
            label = int(labels[i])
            if label == 1:
                continue
            if label not in (0, 2):
                label = 0 if float(labels[i]) <= 0 else 2
            y = 0 if label == 0 else 1
            text = _clean_tensor(texts[i])[:max_text_len]
            audio = _clean_tensor(audios[i])[:max_audio_len]
            vision = _clean_tensor(visions[i])[:max_vision_len]
            self.items.append((str(ids[i]), text, audio, vision, y))

        if not self.items:
            raise ValueError("No binary samples after neutral filtering.")
        self.text_dim = self.items[0][1].shape[-1]
        self.audio_dim = self.items[0][2].shape[-1]
        self.vision_dim = self.items[0][3].shape[-1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _clean_tensor(x, normalize=False):
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr[None, :]
    if normalize:
        # optional per-timestep L2 normalization; DISABLED by default because
        # existing checkpoints (08-28) were trained on raw features. If you
        # re-train with normalize=True, evaluation must use the same setting.
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
    return torch.tensor(arr, dtype=torch.float32)


def collate_batch(batch):
    metas = [b[0] for b in batch]
    text = pad_sequence([b[1] for b in batch], batch_first=True)
    audio = pad_sequence([b[2] for b in batch], batch_first=True)
    vision = pad_sequence([b[3] for b in batch], batch_first=True)
    labels = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return metas, text, audio, vision, labels


def make_negative_batch(text, audio, vision):
    bsz = text.size(0)
    perm = torch.randperm(bsz, device=text.device)
    return text[perm], audio[perm], vision[perm]


class CorMulTReproModel(nn.Module):
    """CorMulT-style correlation-aware MulT with dynamic input dimensions."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        d: int = 30,
        corr_d_model: int = 128,
        corr_out_dim: int = 64,
        num_heads: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
        out_dropout: float = 0.0,
        use_correlation: bool = True,
        corr_model: CorrelationModel | None = None,
        mlp_head: bool = False,
    ):
        super().__init__()
        self.d = d
        self.use_correlation = use_correlation
        self.out_dropout = out_dropout
        self.proj_l = nn.Conv1d(text_dim, d, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, d, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(vision_dim, d, kernel_size=1, bias=False)

        self.trans_l_with_a = self._encoder(d, num_heads, layers, dropout)
        self.trans_l_with_v = self._encoder(d, num_heads, layers, dropout)
        self.trans_a_with_l = self._encoder(d, num_heads, layers, dropout)
        self.trans_a_with_v = self._encoder(d, num_heads, layers, dropout)
        self.trans_v_with_l = self._encoder(d, num_heads, layers, dropout)
        self.trans_v_with_a = self._encoder(d, num_heads, layers, dropout)
        self.trans_l_mem = self._encoder(2 * d, num_heads, 3, dropout)
        self.trans_a_mem = self._encoder(2 * d, num_heads, 3, dropout)
        self.trans_v_mem = self._encoder(2 * d, num_heads, 3, dropout)

        combined_dim = 6 * d
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        if mlp_head:
            # Same-capacity 2-layer MLP head as the proposed model's classifier
            self.out_layer = nn.Sequential(
                nn.LayerNorm(combined_dim),
                nn.Dropout(out_dropout),
                nn.Linear(combined_dim, combined_dim // 2),
                nn.GELU(),
                nn.Dropout(out_dropout / 2),
                nn.Linear(combined_dim // 2, 2),
            )
        else:
            self.out_layer = nn.Linear(combined_dim, 2)

        self.corr_model = corr_model
        if self.use_correlation and self.corr_model is None:
            self.corr_model = CorrelationModel(
                text_in_dim=text_dim,
                audio_in_dim=audio_dim,
                vision_in_dim=vision_dim,
                d_model=corr_d_model,
                num_layers=2,
                num_heads=4,
                dim_feedforward=256,
                dropout=dropout,
                out_dim=corr_out_dim,
            )
        if self.corr_model is not None:
            for p in self.corr_model.parameters():
                p.requires_grad = False
            self.corr_model.eval()

    @staticmethod
    def _encoder(dim, heads, layers, dropout):
        return TransformerEncoder(
            embed_dim=dim,
            num_heads=heads,
            layers=layers,
            attn_dropout=dropout,
            relu_dropout=dropout,
            res_dropout=dropout,
            embed_dropout=dropout,
            attn_mask=True,
        )

    def _corr(self, text, audio, vision):
        if not self.use_correlation or self.corr_model is None:
            bsz = text.size(0)
            z = torch.zeros(bsz, device=text.device)
            return z, z, z
        with torch.no_grad():
            ft, fa, fv = self.corr_model(text, audio, vision)
            ft = ft.mean(dim=1)
            fa = fa.mean(dim=1)
            fv = fv.mean(dim=1)
            c_ta = F.cosine_similarity(ft, fa, dim=-1)
            c_tv = F.cosine_similarity(ft, fv, dim=-1)
            c_av = F.cosine_similarity(fa, fv, dim=-1)
        return c_ta, c_tv, c_av

    def forward(self, text, audio, vision):
        bsz = text.size(0)
        c_ta, c_tv, c_av = self._corr(text, audio, vision)

        xl = F.dropout(text.transpose(1, 2), p=0.1, training=self.training)
        xa = audio.transpose(1, 2)
        xv = vision.transpose(1, 2)
        pl = self.proj_l(xl).permute(2, 0, 1)
        pa = self.proj_a(xa).permute(2, 0, 1)
        pv = self.proj_v(xv).permute(2, 0, 1)

        hl_a = self.trans_l_with_a(pl, pa, pa) * (1.0 + c_ta.view(1, bsz, 1))
        hl_v = self.trans_l_with_v(pl, pv, pv) * (1.0 + c_tv.view(1, bsz, 1))
        ha_l = self.trans_a_with_l(pa, pl, pl) * (1.0 + c_ta.view(1, bsz, 1))
        ha_v = self.trans_a_with_v(pa, pv, pv) * (1.0 + c_av.view(1, bsz, 1))
        hv_l = self.trans_v_with_l(pv, pl, pl) * (1.0 + c_tv.view(1, bsz, 1))
        hv_a = self.trans_v_with_a(pv, pa, pa) * (1.0 + c_av.view(1, bsz, 1))

        hl = self.trans_l_mem(torch.cat([hl_a, hl_v], dim=2))[-1]
        ha = self.trans_a_mem(torch.cat([ha_l, ha_v], dim=2))[-1]
        hv = self.trans_v_mem(torch.cat([hv_l, hv_a], dim=2))[-1]
        h = torch.cat([hl, ha, hv], dim=1)
        hp = self.proj2(F.dropout(F.relu(self.proj1(h)), p=self.out_dropout, training=self.training))
        h = h + hp
        return self.out_layer(h)


def macro_f1(preds, labels):
    vals = []
    for c in [0, 1]:
        tp = sum((p == c and y == c) for p, y in zip(preds, labels))
        fp = sum((p == c and y != c) for p, y in zip(preds, labels))
        fn = sum((p != c and y == c) for p, y in zip(preds, labels))
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        vals.append(2 * pr * rc / max(1e-8, pr + rc))
    return float(np.mean(vals))


def evaluate(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    loss_total, n_total = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _, text, audio, vision, labels in loader:
            text, audio, vision, labels = text.to(device), audio.to(device), vision.to(device), labels.to(device)
            logits = model(text, audio, vision)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=-1)
            preds.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            loss_total += loss.item() * labels.size(0)
            n_total += labels.size(0)
    acc = sum(p == y for p, y in zip(preds, labels_all)) / len(labels_all)
    return loss_total / max(1, n_total), float(acc), macro_f1(preds, labels_all)


def pretrain_correlation(args, train_loader, valid_loader, dims, device, out_dir):
    model = CorrelationModel(
        text_in_dim=dims.text,
        audio_in_dim=dims.audio,
        vision_in_dim=dims.vision,
        d_model=args.corr_d_model,
        num_layers=args.corr_layers,
        num_heads=args.corr_heads,
        dim_feedforward=args.corr_ffn,
        dropout=args.dropout,
        out_dim=args.corr_out_dim,
    ).to(device)
    opt = Adam(model.parameters(), lr=args.corr_lr)
    triple = TripleLoss(margin=args.corr_margin)
    best_state, best_val = None, float("inf")
    for epoch in range(1, args.corr_epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for _, text, audio, vision, _ in train_loader:
            text, audio, vision = text.to(device), audio.to(device), vision.to(device)
            text_n, audio_n, vision_n = make_negative_batch(text, audio, vision)
            opt.zero_grad()
            ft, fa, fv = model(text, audio, vision)
            ft_n, fa_n, fv_n = model(text_n, audio_n, vision_n)
            loss = (
                triple(fa, ft, ft_n) + triple(fa, fv, fv_n)
                + triple(ft, fa, fa_n) + triple(ft, fv, fv_n)
                + triple(fv, fa, fa_n) + triple(fv, ft, ft_n)
            ) / 6.0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            train_loss += loss.item() * text.size(0)
            n += text.size(0)
        val_loss = 0.0
        val_n = 0
        model.eval()
        with torch.no_grad():
            for _, text, audio, vision, _ in valid_loader:
                text, audio, vision = text.to(device), audio.to(device), vision.to(device)
                text_n, audio_n, vision_n = make_negative_batch(text, audio, vision)
                ft, fa, fv = model(text, audio, vision)
                ft_n, fa_n, fv_n = model(text_n, audio_n, vision_n)
                loss = (
                    triple(fa, ft, ft_n) + triple(fa, fv, fv_n)
                    + triple(ft, fa, fa_n) + triple(ft, fv, fv_n)
                    + triple(fv, fa, fa_n) + triple(fv, ft, ft_n)
                ) / 6.0
                val_loss += loss.item() * text.size(0)
                val_n += text.size(0)
        val_loss /= max(1, val_n)
        print(f"[corr] epoch={epoch}/{args.corr_epochs} train={train_loss/max(1,n):.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    path = out_dir / f"{args.dataset}_cormult_corr_seed{args.seed}.pt"
    torch.save(model.state_dict(), path)
    print(f"saved correlation model: {path}", flush=True)
    return model, path


def train_classifier(args, train_loader, valid_loader, test_loader, dims, device, corr_model, out_dir):
    model = CorMulTReproModel(
        text_dim=dims.text,
        audio_dim=dims.audio,
        vision_dim=dims.vision,
        d=args.d,
        corr_d_model=args.corr_d_model,
        corr_out_dim=args.corr_out_dim,
        num_heads=args.num_heads,
        layers=args.layers,
        dropout=args.dropout,
        out_dropout=args.out_dropout,
        use_correlation=args.use_correlation,
        corr_model=corr_model,
        mlp_head=getattr(args, "mlp_head", False),
    ).to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(opt, mode="min", patience=args.when, factor=0.5)
    best_state, best_val_f1 = None, -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for _, text, audio, vision, labels in train_loader:
            text, audio, vision, labels = text.to(device), audio.to(device), vision.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(text, audio, vision)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            loss_sum += loss.item() * labels.size(0)
            n += labels.size(0)
        val_loss, val_acc, val_f1 = evaluate(model, valid_loader, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
        scheduler.step(val_loss)
        print(
            f"[cls] epoch={epoch}/{args.epochs} train={loss_sum/max(1,n):.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_acc, val_f1 = evaluate(model, valid_loader, device)
    _, test_acc, test_f1 = evaluate(model, test_loader, device)
    suffix = "_mlp" if getattr(args, "mlp_head", False) else ""
    ckpt = out_dir / f"{args.dataset}_cormult_repro{suffix}_seed{args.seed}.pt"
    torch.save(model.state_dict(), ckpt)
    return model, ckpt, val_acc, val_f1, test_acc, test_f1


def main():
    parser = argparse.ArgumentParser(description="CorMulT reproduction on unified binary MOSI/MOSEI/SIMS pkl files.")
    parser.add_argument("--dataset", choices=["MOSI", "MOSEI", "SIMS"], required=True)
    parser.add_argument("--data_root", default=os.getenv("MSA_DATA_ROOT"))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--corr_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corr_lr", type=float, default=2e-4)
    parser.add_argument("--when", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.8)
    parser.add_argument("--d", type=int, default=30)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dropout", type=float, default=0.0)
    parser.add_argument("--corr_d_model", type=int, default=128)
    parser.add_argument("--corr_layers", type=int, default=2)
    parser.add_argument("--corr_heads", type=int, default=4)
    parser.add_argument("--corr_ffn", type=int, default=256)
    parser.add_argument("--corr_out_dim", type=int, default=64)
    parser.add_argument("--corr_margin", type=float, default=0.2)
    parser.add_argument("--max_text_len", type=int, default=50)
    parser.add_argument("--max_audio_len", type=int, default=100)
    parser.add_argument("--max_vision_len", type=int, default=100)
    parser.add_argument("--use_correlation", action="store_true")
    parser.add_argument("--mlp_head", action="store_true",
                        help="Replace the single linear head with a 2-layer MLP head (same capacity as the proposed model).")
    parser.add_argument("--skip_corr_pretrain", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir or f"results_cormult_repro/{args.dataset}/seed_{args.seed}").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pickle.load(open(resolve_dataset_pkl(args.dataset, args.data_root), "rb"))
    train_ds = VWBSequenceDataset(raw["train"], args.max_text_len, args.max_audio_len, args.max_vision_len)
    valid_ds = VWBSequenceDataset(raw["valid"], args.max_text_len, args.max_audio_len, args.max_vision_len)
    test_ds = VWBSequenceDataset(raw["test"], args.max_text_len, args.max_audio_len, args.max_vision_len)
    dims = SimpleNamespace(text=train_ds.text_dim, audio=train_ds.audio_dim, vision=train_ds.vision_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch * 2, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False, collate_fn=collate_batch, num_workers=0)
    print(
        f"dataset={args.dataset} train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} "
        f"dims=({dims.text},{dims.audio},{dims.vision}) device={device} use_correlation={args.use_correlation}",
        flush=True,
    )

    corr_model, corr_path = None, ""
    if args.use_correlation and not args.skip_corr_pretrain:
        corr_model, corr_path = pretrain_correlation(args, train_loader, valid_loader, dims, device, out_dir)

    _, ckpt, val_acc, val_f1, test_acc, test_f1 = train_classifier(
        args, train_loader, valid_loader, test_loader, dims, device, corr_model, out_dir
    )
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.dataset}_cormult_repro_result_{ts}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "method", "seed", "use_correlation", "val_acc", "val_f1",
                "test_acc", "test_f1", "ckpt", "corr_ckpt",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "method": "CorMulT" if args.use_correlation else "MulT",
            "seed": args.seed,
            "use_correlation": args.use_correlation,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "ckpt": str(ckpt),
            "corr_ckpt": str(corr_path),
        })
    print(f"FINAL val_acc={val_acc:.4f} val_f1={val_f1:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
