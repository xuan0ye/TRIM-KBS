"""
端到端多模态情感分析训练（解冻骨干）
────────────────────────────────────────────────────────────────
步骤一  数据加载：从 PKL 读取 raw text_bert tokens + 预提取音视觉特征
步骤二  模型：BERT 编码器（文本）+ MLP 投影器（音视觉）+ 层次融合头
步骤三  分级解冻：Phase-1 只训练头 → Phase-2 解冻 BERT 后 N 层 → Phase-3 全解冻
步骤四  分层学习率：backbone lr_b=2e-5 / projector lr_p=5e-5 / fusion head lr_h=3e-4
步骤五  训练输出：每轮打印 loss / val_acc / macro_f1，保存最优权重和报告

用法:
  python run_e2e_msa.py --dataset MOSI
  python run_e2e_msa.py --dataset SIMS  --bert_model bert-base-chinese
  python run_e2e_msa.py --dataset MOSEI --max_train 4000
"""
import io, sys
try:
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse, copy, csv, os, pickle, random, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertModel, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
class GatedCrossModalBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.gate       = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.norm1      = nn.LayerNorm(d)
        self.ff         = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.norm2      = nn.LayerNorm(d)
        self.drop       = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_val: torch.Tensor) -> torch.Tensor:
        """query/key_val: [B, L, D]"""
        attn_out, _ = self.cross_attn(query, key_val, key_val)
        gate = self.gate(torch.cat([query, attn_out], dim=-1))
        h = self.norm1(query + self.drop(gate * attn_out))
        h = self.norm2(h + self.drop(self.ff(h)))
        return h

DEFAULT_DATASET_REL = {
    "MOSI":  Path("MOSI/Processed/aligned_50.pkl"),
    "MOSEI": Path("MOSEI/Processed/aligned_50.pkl"),
    "SIMS":  Path("SIMS/Processed/unaligned_39.pkl"),
}
BERT_DEFAULT = {
    "MOSI":  "bert-base-uncased",
    "MOSEI": "bert-base-uncased",
    "SIMS":  "bert-base-chinese",
}

E2E_ABLATION_PRESETS = {
    "full": {},
    "no_uni_ctx": {"use_uni_ctx": False},
    "no_mm_ctx": {"use_mm_ctx": False},
    "no_summary_token": {"use_summary_token": False},
    "no_contrast": {"use_contrastive": False},
    "no_ot": {"use_ot": False},
    "no_ama": {"use_ama": False},
    "no_aux": {"use_contrastive": False, "use_ot": False, "use_ama": False},
    "no_modrop": {"modality_dropout_p": 0.0},
    "no_missing_repr": {"use_missing_repr": False},
    "no_gcm": {"use_gcm": False},
    "text_only": {"use_audio": False, "use_visual": False, "modality_dropout_p": 0.0},
}


def resolve_dataset_pkl(dataset: str, data_root: str | None) -> Path:
    roots = []
    if data_root:
        roots.append(Path(data_root))
    env_root = os.getenv("MSA_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path("g:/VWB/MSA Datasets"))
    rel = DEFAULT_DATASET_REL[dataset]
    for r in roots:
        p = r / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"找不到数据集文件：{dataset}。\n"
        f"请传 --data_root 或设置环境变量 MSA_DATA_ROOT。"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 步骤一：数据加载器
# ──────────────────────────────────────────────────────────────────────────────

class MSARawDataset(Dataset):
    """
    从 PKL 加载：
      - text: (3, seq_len) int64  [input_ids, attn_mask, token_type_ids]
      - audio: (d_a,) float32   预提取均值特征（保留用于 MLP 投影）
      - vision:(d_v,) float32   预提取均值特征（保留用于 MLP 投影）
      - label: int
    """
    def __init__(self, split_data: dict, label_key="classification_labels"):
        ids     = split_data.get("id", [])
        bert    = split_data.get("text_bert",  None)   # [N, 3, seq_len]
        audios  = split_data.get("audio",      None)   # [N, T, d_a]
        visions = split_data.get("vision",     None)   # [N, T, d_v]
        labels  = split_data.get(label_key,
                  split_data.get("regression_labels", None))

        self.items = []
        n = len(ids)
        for i in range(n):
            lab = int(labels[i]) if labels is not None else 0
            if not (0 <= lab <= 2): lab = 1

            # text token ids: [3, seq_len]
            t = torch.tensor(bert[i], dtype=torch.long) if bert is not None \
                else torch.zeros(3, 50, dtype=torch.long)

            # audio mean-pool → [d_a]
            a = torch.tensor(audios[i].mean(0), dtype=torch.float32) \
                if audios is not None else torch.zeros(1)

            # vision mean-pool → [d_v]
            v = torch.tensor(visions[i].mean(0), dtype=torch.float32) \
                if visions is not None else torch.zeros(1)

            self.items.append((t, a, v, lab))

        self.a_dim = audios.shape[2]  if audios  is not None else 1
        self.v_dim = visions.shape[2] if visions is not None else 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_binary_dataset(ds: MSARawDataset) -> MSARawDataset:
    """过滤 neutral(1)，neg(0)→0, pos(2)→1"""
    filtered = [(t, a, v, 0 if l==0 else 1)
                for (t, a, v, l) in ds.items if l != 1]
    new_ds = copy.copy(ds)
    new_ds.items = filtered
    return new_ds


def make_loaders(raw: dict, binary: bool, batch_size: int,
                 seed: int = 42, max_train: int = None):
    tr_ds = MSARawDataset(raw["train"])
    va_ds = MSARawDataset(raw["valid"])
    te_ds = MSARawDataset(raw["test"])
    if binary:
        tr_ds = build_binary_dataset(tr_ds)
        va_ds = build_binary_dataset(va_ds)
        te_ds = build_binary_dataset(te_ds)
    if max_train and len(tr_ds) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tr_ds), max_train, replace=False).tolist()
        tr_ds_sub = copy.copy(tr_ds)
        tr_ds_sub.items = [tr_ds.items[i] for i in idx]
        tr_ds = tr_ds_sub
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    return tr_loader, va_loader, te_loader, tr_ds.a_dim, tr_ds.v_dim


# ──────────────────────────────────────────────────────────────────────────────
# 步骤二：端到端模型（BERT + MLP 投影器 + 层次融合头）
# ──────────────────────────────────────────────────────────────────────────────

class E2EMSAModel(nn.Module):
    """
    文本：BertModel → [CLS] (768d)
    音频：Linear(d_a → d_model) + GELU + Linear(d_model → d_model)
    视觉：Linear(d_v → d_model) + GELU + Linear(d_model → d_model)
    融合：单模态上下文(可选) → 门控跨模态(GCM) → 多模态上下文(摘要Token) → 分类头
    """

    def __init__(self, bert_model_name: str, a_dim: int, v_dim: int,
                 d_model: int = 256, n_classes: int = 2,
                 uni_layers: int = 2, uni_heads: int = 4,
                 mm_layers: int = 2, mm_heads: int = 8, n_summary: int = 3,
                 dropout: float = 0.1, classifier_dropout: float = 0.4,
                 modality_dropout_p: float = 0.15,
                 use_uni_ctx: bool = True, use_gcm: bool = True,
                 use_mm_ctx: bool = True, use_contrastive: bool = True,
                 use_ot: bool = True, use_ama: bool = True,
                 use_missing_repr: bool = True,
                 use_summary_token: bool = True,
                 use_audio: bool = True, use_visual: bool = True):
        super().__init__()
        self.d = d_model
        self.modality_dropout_p = modality_dropout_p
        self.n_summary = n_summary
        self.use_uni_ctx = use_uni_ctx
        self.use_gcm = use_gcm
        self.use_mm_ctx = use_mm_ctx
        self.use_contrastive = use_contrastive
        self.use_ot = use_ot
        self.use_ama = use_ama
        self.use_missing_repr = use_missing_repr
        self.use_summary_token = use_summary_token
        self.use_audio = use_audio
        self.use_visual = use_visual

        # ── 文本骨干 ──────────────────────────────────────────────────────────
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.proj_t = nn.Linear(768, d_model)

        # ── 音视觉 MLP 投影器（含可学习层，保留梯度路径）────────────────────
        def _projector(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim * 2),
                nn.LayerNorm(out_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim),
            )
        self.proj_a = _projector(a_dim, d_model)
        self.proj_v = _projector(v_dim, d_model)

        # ── 缺失模态可学习占位 ─────────────────────────────────────────────
        self.missing_t = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.missing_a = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.missing_v = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # ── 层次融合：单模态上下文 ─────────────────────────────────────────
        enc_layer = lambda h: nn.TransformerEncoderLayer(
            d_model=d_model, nhead=h, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.uni_t = nn.TransformerEncoder(enc_layer(uni_heads), uni_layers)
        self.uni_a = nn.TransformerEncoder(enc_layer(uni_heads), uni_layers)
        self.uni_v = nn.TransformerEncoder(enc_layer(uni_heads), uni_layers)

        # ── 门控跨模态 ────────────────────────────────────────────────────
        self.gcm_ta = GatedCrossModalBlock(d_model, n_heads=mm_heads)
        self.gcm_at = GatedCrossModalBlock(d_model, n_heads=mm_heads)
        self.gcm_tv = GatedCrossModalBlock(d_model, n_heads=mm_heads)

        # ── 多模态上下文（摘要 Token）──────────────────────────────────────
        self.mm_summary = nn.Parameter(torch.randn(n_summary, d_model) * 0.02)
        self.mm_ctx     = nn.TransformerEncoder(enc_layer(mm_heads), mm_layers)

        # ── 对比投影头 ────────────────────────────────────────────────────
        self.cproj_t = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128))
        self.cproj_a = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128))
        self.cproj_v = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128))

        # ── 最优传输线性变换 ───────────────────────────────────────────────
        self.ot_ta = nn.Linear(d_model, d_model)
        self.ot_tv = nn.Linear(d_model, d_model)

        # ── AMA 参数 ───────────────────────────────────────────────────────
        self.task_log_vars = nn.Parameter(torch.zeros(3))
        self.task_priority = nn.Parameter(torch.ones(3) / 3)

        # ── 分类头 ────────────────────────────────────────────────────────
        in_dim = d_model * n_summary if (use_mm_ctx and use_summary_token) else d_model * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(classifier_dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout / 2),
            nn.Linear(in_dim // 2, n_classes),
        )

    # ── 辅助损失 ─────────────────────────────────────────────────────────────
    def _sym_nce(self, a, b, temp=0.1):
        a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
        N = a.size(0)
        if N < 2: return torch.tensor(0., device=a.device)
        logits = a @ b.T / temp
        tgt = torch.arange(N, device=a.device)
        return (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2

    def _sinkhorn(self, a, b, reg=0.5, n_iter=5):
        if a.size(0) < 2: return torch.tensor(0., device=a.device)
        an = F.normalize(self.ot_ta(a), dim=-1)
        bn = F.normalize(b, dim=-1)
        M  = (1.0 - (an @ bn.T)).clamp(0, 2)
        n, m = M.shape
        log_r = torch.zeros(n, device=M.device)
        log_c = torch.zeros(m, device=M.device)
        log_K = -M / reg
        for _ in range(n_iter):
            log_r = -torch.logsumexp(log_K + log_c.unsqueeze(0), dim=1)
            log_c = -torch.logsumexp(log_K + log_r.unsqueeze(1), dim=0)
        return (torch.exp(log_K + log_r.unsqueeze(1) + log_c.unsqueeze(0)) * M).sum()

    def _ama_weight(self, i):
        prio = F.softmax(self.task_priority * 5, dim=0) * 0.5 + \
               torch.ones(3, device=self.task_priority.device) / 3 * 0.5
        prec = torch.exp(-self.task_log_vars)
        return (prec * prio)[i]

    # ── 前向传播 ─────────────────────────────────────────────────────────────
    def forward(self, input_ids, attn_mask, token_type_ids,
                feat_a, feat_v, training=False):
        B = input_ids.size(0)

        # 文本：BERT [CLS]
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attn_mask,
                             token_type_ids=token_type_ids,
                             return_dict=True)
        ht = self.proj_t(bert_out.last_hidden_state[:, 0, :])  # [B, d]

        # 音视觉投影
        ha = self.proj_a(feat_a)   # [B, d]
        hv = self.proj_v(feat_v)   # [B, d]

        # 缺失模态补零 + 模态 dropout（训练时）
        mt = torch.ones(B, device=ht.device)
        ma = torch.ones(B, device=ha.device)
        mv = torch.ones(B, device=hv.device)
        if not self.use_audio:
            ma.zero_()
        if not self.use_visual:
            mv.zero_()
        if training and self.modality_dropout_p > 0:
            for m_ in [ma, mv]:          # 文本不丢
                drop = torch.rand(B, device=m_.device) < self.modality_dropout_p
                m_[drop] = 0.
        if self.use_missing_repr:
            mt_fill = self.missing_t.expand(B, -1)
            ma_fill = self.missing_a.expand(B, -1)
            mv_fill = self.missing_v.expand(B, -1)
        else:
            mt_fill = torch.zeros_like(ht)
            ma_fill = torch.zeros_like(ha)
            mv_fill = torch.zeros_like(hv)
        ht = torch.where(mt.unsqueeze(-1) > 0.5, ht, mt_fill)
        ha = torch.where(ma.unsqueeze(-1) > 0.5, ha, ma_fill)
        hv = torch.where(mv.unsqueeze(-1) > 0.5, hv, mv_fill)

        # 单模态上下文 ([B,1,d])
        if self.use_uni_ctx:
            ht_s = self.uni_t(ht.unsqueeze(1))
            ha_s = self.uni_a(ha.unsqueeze(1))
            hv_s = self.uni_v(hv.unsqueeze(1))
        else:
            ht_s = ht.unsqueeze(1)
            ha_s = ha.unsqueeze(1)
            hv_s = hv.unsqueeze(1)

        # 对比 & OT 辅助损失（使用单模态上下文输出）
        zero = torch.tensor(0., device=ht.device)
        if self.use_contrastive:
            zt = F.normalize(self.cproj_t(ht_s[:, 0]), dim=-1)
            za = F.normalize(self.cproj_a(ha_s[:, 0]), dim=-1)
            zv = F.normalize(self.cproj_v(hv_s[:, 0]), dim=-1)
            gcl = self._sym_nce(zt, za) + self._sym_nce(zt, zv)
        else:
            gcl = zero
        if self.use_ot:
            ot = self._sinkhorn(ht_s[:, 0], ha_s[:, 0]) + \
                 self._sinkhorn(ht_s[:, 0], hv_s[:, 0])
        else:
            ot = zero

        # 门控跨模态
        if self.use_gcm:
            ht2 = self.gcm_ta(ht_s, ha_s)   # [B,1,d]
            ha2 = self.gcm_at(ha_s, ht_s)
            hv2 = self.gcm_tv(hv_s, ht_s)
        else:
            ht2, ha2, hv2 = ht_s, ha_s, hv_s

        # 多模态上下文（摘要 Token 拼接三模态）
        if self.use_mm_ctx:
            if self.use_summary_token:
                summ = self.mm_summary.unsqueeze(0).expand(B, -1, -1)
                seq  = torch.cat([summ, ht2, ha2, hv2], dim=1)
                seq  = self.mm_ctx(seq)
                fused = seq[:, :self.n_summary, :].reshape(B, -1)
            else:
                seq  = torch.cat([ht2, ha2, hv2], dim=1)
                seq  = self.mm_ctx(seq)
                fused = seq.reshape(B, -1)
        else:
            fused = torch.cat([ht2[:, 0], ha2[:, 0], hv2[:, 0]], dim=-1)

        logits = self.classifier(fused)

        w_gcl = self._ama_weight(1)
        w_ot  = self._ama_weight(2)
        reg   = 0.5 * self.task_log_vars.sum() if self.use_ama else zero
        return logits, {"gcl": gcl, "ot": ot,
                        "w_gcl": w_gcl, "w_ot": w_ot, "reg_ama": reg}


# ──────────────────────────────────────────────────────────────────────────────
# 步骤三 & 四：分级解冻 + 分层学习率
# ──────────────────────────────────────────────────────────────────────────────

def freeze_bert_all(model):
    for p in model.bert.parameters():
        p.requires_grad = False

def unfreeze_bert_last_n(model, n: int, verbose=True):
    """解冻 BERT 最后 n 个 Transformer 层 + pooler"""
    for p in model.bert.parameters():
        p.requires_grad = False
    layers = model.bert.encoder.layer
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.bert.pooler.parameters():
        p.requires_grad = True
    if verbose:
        total = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
        print(f"  [解冻] BERT 最后 {n} 层  可训练参数: {total:,}", flush=True)

def unfreeze_bert_all(model, verbose=True):
    for p in model.bert.parameters():
        p.requires_grad = True
    if verbose:
        total = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
        print(f"  [解冻] BERT 全部层  可训练参数: {total:,}", flush=True)

def build_optimizer(model, lr_bert, lr_proj, lr_head, weight_decay=1e-2):
    """分层参数组：骨干 / 投影器 / 融合头"""
    bert_params  = [p for p in model.bert.parameters()  if p.requires_grad]
    proj_params  = (list(model.proj_t.parameters()) +
                    list(model.proj_a.parameters()) +
                    list(model.proj_v.parameters()))
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad
                    and not n.startswith("bert.")
                    and not n.startswith("proj_t.")
                    and not n.startswith("proj_a.")
                    and not n.startswith("proj_v.")]
    groups = []
    if bert_params:
        groups.append({"params": bert_params, "lr": lr_bert,
                        "weight_decay": weight_decay})
    if proj_params:
        groups.append({"params": proj_params, "lr": lr_proj,
                        "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": lr_head,
                        "weight_decay": weight_decay})
    return AdamW(groups)


# ──────────────────────────────────────────────────────────────────────────────
# 步骤五：训练 & 评估
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for tb, ta, tv, lbl in loader:
            # tb=[B,3,50], ta=[B,d_a], tv=[B,d_v]
            tb, ta, tv = tb.to(device), ta.to(device), tv.to(device)
            logits, _ = model(tb[:, 0], tb[:, 1], tb[:, 2], ta, tv)
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(lbl.tolist())
    n_cls = max(gts) + 1
    f1s = []
    for c in range(n_cls):
        tp = sum((p==c and g==c) for p, g in zip(preds, gts))
        fp = sum((p==c and g!=c) for p, g in zip(preds, gts))
        fn = sum((p!=c and g==c) for p, g in zip(preds, gts))
        pr = tp / max(1, tp+fp); rc = tp / max(1, tp+fn)
        f1s.append(2*pr*rc / max(1e-8, pr+rc))
    acc = sum(p==g for p, g in zip(preds, gts)) / len(gts)
    return acc, float(np.mean(f1s))


def train_phase(model, tr_loader, va_loader, device, n_classes,
                tr_labels, optimizer, epochs, phase_name,
                acc_steps=4):
    total_steps  = max(1, len(tr_loader) * epochs // acc_steps)
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    counts = torch.bincount(torch.tensor(tr_labels, dtype=torch.long),
                            minlength=n_classes).float()
    raw_w  = torch.sqrt(counts.sum() / (n_classes * counts.clamp(min=1)))
    class_w = (raw_w / raw_w.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)

    best_f1, best_acc, best_state = 0., 0., None

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.; step = 0; optimizer.zero_grad()
        for i, (tb, ta, tv, lbl) in enumerate(tr_loader):
            # tb=[B,3,50]  ta=[B,d_a]  tv=[B,d_v]
            tb, ta, tv, lbl = (tb.to(device), ta.to(device),
                                tv.to(device), lbl.to(device))
            logits, aux = model(tb[:, 0], tb[:, 1], tb[:, 2], ta, tv, training=True)
            loss = criterion(logits, lbl)
            loss = loss + 0.02 * aux["gcl"] + 0.02 * aux["ot"] + 0.001 * aux["reg_ama"]
            loss = loss / acc_steps
            loss.backward()
            if (i + 1) % acc_steps == 0 or (i + 1) == len(tr_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                step += 1
            ep_loss += loss.item() * acc_steps

        val_acc, val_f1 = evaluate(model, va_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1; best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        avg = ep_loss / max(1, len(tr_loader))
        print(f"  [{phase_name}] ep{epoch}/{epochs}  loss={avg:.4f}"
              f"  acc={val_acc:.4f}  f1={val_f1:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    return best_acc, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["MOSI","MOSEI","SIMS"], default="MOSI")
    parser.add_argument("--data_root", type=str, default=None,
                        help="MSA Datasets 根目录（例如 /home/user/MSA Datasets）")
    parser.add_argument("--output_root", type=str, default=None,
                        help="结果输出根目录，默认当前工程下 results_e2e_*")
    parser.add_argument("--bert_model",type=str, default=None)
    parser.add_argument("--out_dir",   default=None)
    parser.add_argument("--d_model",   type=int,   default=256)
    # 三个训练阶段轮数
    parser.add_argument("--phase1_ep", type=int,   default=5,
                        help="阶段1：冻结BERT，只训练投影+融合头")
    parser.add_argument("--phase2_ep", type=int,   default=8,
                        help="阶段2：解冻BERT后N层")
    parser.add_argument("--phase3_ep", type=int,   default=7,
                        help="阶段3：全解冻BERT")
    parser.add_argument("--unfreeze_n",type=int,   default=2,
                        help="阶段2解冻BERT最后N层")
    # 学习率
    parser.add_argument("--lr_bert",   type=float, default=2e-5)
    parser.add_argument("--lr_proj",   type=float, default=5e-5)
    parser.add_argument("--lr_head",   type=float, default=3e-4)
    # 其他
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--acc_steps", type=int,   default=4,
                        help="梯度累积步数（等效 batch=batch×acc_steps）")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--max_train", type=int,   default=None)
    parser.add_argument("--binary",    action="store_true", default=True)
    parser.add_argument("--ablation", choices=sorted(E2E_ABLATION_PRESETS),
                        default="full",
                        help="端到端消融预设；默认 full 保持原完整模型")
    args = parser.parse_args()

    bert_name = args.bert_model or BERT_DEFAULT[args.dataset]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.output_root:
        out_dir = Path(args.output_root) / f"results_e2e_{args.dataset.lower()}"
    else:
        out_dir = Path(f"results_e2e_{args.dataset.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    pkl_path = resolve_dataset_pkl(args.dataset, args.data_root)
    print(f"加载 {pkl_path}…", flush=True)
    raw = pickle.load(open(pkl_path, "rb"))

    tr_loader, va_loader, te_loader, a_dim, v_dim = make_loaders(
        raw, binary=args.binary, batch_size=args.batch,
        seed=args.seed, max_train=args.max_train)
    n_classes = 2 if args.binary else 3
    tr_labels = np.array([item[3] for item in tr_loader.dataset.items])

    print(f"数据集: {args.dataset}  训练: {len(tr_loader.dataset)}"
          f"  验证: {len(va_loader.dataset)}  测试: {len(te_loader.dataset)}"
          f"  n_classes={n_classes}", flush=True)
    print(f"音频维度: {a_dim}  视觉维度: {v_dim}", flush=True)
    print(f"BERT: {bert_name}", flush=True)
    print(f"端到端消融: {args.ablation}", flush=True)
    eff_batch = args.batch * args.acc_steps
    print(f"batch={args.batch}  acc_steps={args.acc_steps}  "
          f"等效batch={eff_batch}", flush=True)

    cls_cnt = np.bincount(tr_labels, minlength=n_classes)
    for c, cnt in enumerate(cls_cnt):
        print(f"  类别{c}: {cnt} ({cnt/len(tr_labels)*100:.1f}%)", flush=True)

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    print(f"\n加载 BERT: {bert_name}…", flush=True)
    ablation_cfg = E2E_ABLATION_PRESETS[args.ablation]
    model = E2EMSAModel(bert_name, a_dim=a_dim, v_dim=v_dim,
                        d_model=args.d_model, n_classes=n_classes,
                        **ablation_cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_p:,}", flush=True)

    # ── Phase 1：冻结 BERT，只训练投影器 + 融合头 ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase-1: 冻结 BERT，训练投影器 + 融合头  ({args.phase1_ep} 轮)")
    print(f"{'='*60}", flush=True)
    freeze_bert_all(model)
    opt1 = build_optimizer(model, lr_bert=0, lr_proj=args.lr_head,
                            lr_head=args.lr_head)
    p1_acc, p1_f1 = train_phase(model, tr_loader, va_loader, device,
                                  n_classes, tr_labels, opt1,
                                  args.phase1_ep, "P1-FrozenBERT", args.acc_steps)
    print(f"  Phase-1 最佳: acc={p1_acc:.4f}  f1={p1_f1:.4f}", flush=True)

    # ── Phase 2：解冻 BERT 最后 N 层 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase-2: 解冻 BERT 最后 {args.unfreeze_n} 层  ({args.phase2_ep} 轮)")
    print(f"  lr_bert={args.lr_bert}  lr_proj={args.lr_proj}  lr_head={args.lr_head}")
    print(f"{'='*60}", flush=True)
    unfreeze_bert_last_n(model, args.unfreeze_n)
    opt2 = build_optimizer(model, lr_bert=args.lr_bert,
                            lr_proj=args.lr_proj, lr_head=args.lr_head)
    p2_acc, p2_f1 = train_phase(model, tr_loader, va_loader, device,
                                  n_classes, tr_labels, opt2,
                                  args.phase2_ep, "P2-UnfreezeLast", args.acc_steps)
    print(f"  Phase-2 最佳: acc={p2_acc:.4f}  f1={p2_f1:.4f}", flush=True)

    # ── Phase 3：全解冻 BERT ──────────────────────────────────────────────────
    if args.phase3_ep > 0:
        print(f"\n{'='*60}")
        print(f"  Phase-3: 全解冻 BERT  ({args.phase3_ep} 轮)")
        print(f"  lr_bert={args.lr_bert/2:.1e}  lr_proj={args.lr_proj}  lr_head={args.lr_head}")
        print(f"{'='*60}", flush=True)
        unfreeze_bert_all(model)
        opt3 = build_optimizer(model, lr_bert=args.lr_bert / 2,
                                lr_proj=args.lr_proj, lr_head=args.lr_head)
        p3_acc, p3_f1 = train_phase(model, tr_loader, va_loader, device,
                                      n_classes, tr_labels, opt3,
                                      args.phase3_ep, "P3-FullUnfreeze", args.acc_steps)
        print(f"  Phase-3 最佳: acc={p3_acc:.4f}  f1={p3_f1:.4f}", flush=True)
    else:
        p3_acc, p3_f1 = p2_acc, p2_f1

    best_acc = max(p1_acc, p2_acc, p3_acc)
    best_f1  = max(p1_f1,  p2_f1,  p3_f1)
    test_acc, test_f1 = evaluate(model, te_loader, device)
    print(f"\n测试集: acc={test_acc:.4f}  f1={test_f1:.4f}", flush=True)

    # ── 保存模型 & 报告 ───────────────────────────────────────────────────────
    ckpt_path = out_dir / f"{args.dataset}_e2e_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n模型已保存: {ckpt_path}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    rpt = [
        "=" * 68,
        f"  {args.dataset} 端到端训练报告（解冻骨干）",
        f"  BERT: {bert_name}  a_dim={a_dim}  v_dim={v_dim}",
        f"  d_model={args.d_model}  n_classes={n_classes}",
        f"  ablation={args.ablation}",
        f"  batch={args.batch}×acc{args.acc_steps}={eff_batch}",
        f"  lr_bert={args.lr_bert}  lr_proj={args.lr_proj}  lr_head={args.lr_head}",
        "=" * 68, "",
        f"  {'阶段':<20} {'BestAcc':>9} {'MacroF1':>9}",
        "  " + "-" * 44,
        f"  {'Phase1 冻结BERT':<20} {p1_acc:>9.4f} {p1_f1:>9.4f}",
        f"  {'Phase2 解冻后'+str(args.unfreeze_n)+'层':<20} {p2_acc:>9.4f} {p2_f1:>9.4f}",
        f"  {'Phase3 全解冻BERT':<20} {p3_acc:>9.4f} {p3_f1:>9.4f}",
        "  " + "-" * 44,
        f"  {'最终最优':<20} {best_acc:>9.4f} {best_f1:>9.4f}",
        f"  {'测试集':<20} {test_acc:>9.4f} {test_f1:>9.4f}",
        "",
        f"  对比离线特征基线 (MacroF1):",
        f"    MOSI  offline=0.8045  → 目标超越",
        f"    SIMS  offline=0.5448  → 目标超越",
        f"    MOSEI offline=0.8196  → 目标超越",
    ]
    rpt_path = out_dir / f"{args.dataset}_e2e_report_{ts}.txt"
    rpt_path.write_text("\n".join(rpt), encoding="utf-8")
    print("\n" + "\n".join(rpt), flush=True)
    print(f"\n报告已保存: {rpt_path}", flush=True)

    csv_path = out_dir / f"{args.dataset}_e2e_result_{ts}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "bert_model", "ablation", "seed", "binary",
                "phase1_acc", "phase1_f1", "phase2_acc", "phase2_f1",
                "phase3_acc", "phase3_f1", "best_val_acc", "best_val_f1",
                "test_acc", "test_f1", "ckpt_path",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "bert_model": bert_name,
            "ablation": args.ablation,
            "seed": args.seed,
            "binary": args.binary,
            "phase1_acc": p1_acc,
            "phase1_f1": p1_f1,
            "phase2_acc": p2_acc,
            "phase2_f1": p2_f1,
            "phase3_acc": p3_acc,
            "phase3_f1": p3_f1,
            "best_val_acc": best_acc,
            "best_val_f1": best_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "ckpt_path": str(ckpt_path),
        })
    print(f"CSV已保存: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
