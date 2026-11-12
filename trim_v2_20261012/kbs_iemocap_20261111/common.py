from pathlib import Path
import csv, hashlib, json, os, random, sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))

def dump(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, p)

def csvout(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def npz(path, **items):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **items)
    os.replace(tmp, p)

def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}

def verify_code():
    path = HERE / 'bundle_manifest.json'
    if not path.exists(): raise RuntimeError('Build the source manifest first.')
    for name, digest in read(path)['files'].items():
        if sha(HERE / name) != digest: raise RuntimeError('Source changed; build a new package: ' + name)
    return sha(path)

def plan(): return read(HERE / 'protocol.json')

def metrics(y, logits):
    y = np.asarray(y, dtype=np.int64); z = np.asarray(logits, dtype=np.float64)
    if z.shape != (len(y), 6) or not len(y): raise ValueError('Expected nonempty N x 6 logits')
    if not np.isfinite(z).all() or not np.isin(y, np.arange(6)).all(): raise ValueError('Invalid logits/labels')
    q = z.argmax(-1); c = np.bincount(6 * y + q, minlength=36).reshape(6, 6)
    tp = c.diagonal(); support = c.sum(1)
    f1 = 2 * tp / np.maximum(1, support + c.sum(0))
    recall = tp / np.maximum(1, support)
    zz = z - z.max(1, keepdims=True); logp = zz - np.log(np.exp(zz).sum(1, keepdims=True))
    prob = np.exp(logp); truth = np.eye(6)[y]
    return dict(n=len(y), accuracy=float((q == y).mean()), macro_f1=float(f1.mean()),
                weighted_f1=float((f1 * support).sum() / len(y)), uar=float(recall.mean()),
                nll=float(-logp[np.arange(len(y)), y].mean()),
                brier=float(np.square(prob - truth).sum(1).mean()))

def runtime(device='cuda:0', seed=0):
    import torch
    torch.set_num_threads(4); torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; activate wjhenv.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return torch.device(device)

def selection_data(out, dataset, role):
    if role not in ['fit', 'dev']: raise RuntimeError('Training may only read fit/dev arrays.')
    p = Path(out) / 'data' / dataset / f'{role}.npz'; meta = read(p.parent / 'prepared.json')
    if sha(p) != meta['files'][p.name]: raise RuntimeError('Prepared data changed: ' + str(p))
    return arrays(p)

def evaluation_data(out, dataset, role):
    if role != 'test': raise ValueError(role)
    seal = read(Path(out) / 'training_seal.json')
    if seal['protocol_sha256'] != sha(HERE / 'protocol.json') or seal['bundle_sha256'] != verify_code():
        raise RuntimeError('Training seal does not match source and protocol.')
    expected_jobs = len(plan()['variants']) * len(plan()['seeds'])
    if seal['jobs'] != expected_jobs or len(seal['checkpoint_sha256']) != expected_jobs:
        raise RuntimeError('All planned training jobs must be sealed before evaluation.')
    p = Path(out) / 'data' / dataset / 'test.npz'; meta = read(p.parent / 'prepared.json')
    if sha(p) != meta['files'][p.name]: raise RuntimeError('Test arrays changed')
    return arrays(p)
