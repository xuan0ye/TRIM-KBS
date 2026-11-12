"""Validation-only audio representation pilot around the archived paper model."""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
HERE = Path(__file__).resolve().parent
CORE = HERE / '_paper'
sys.path.insert(0, str(CORE))
from common import sha_file, dump_json, write_csv, prepare, read_raw
GROUPS = ('paper_original', 'paper_normalized', 'panns_projected', 'panns_shuffled')
SPLITS = ('train', 'valid')
MASKS = {'T': (1,0,0), 'TA': (1,1,0), 'TV': (1,0,1), 'TAV': (1,1,1)}
DATA_SHA = '45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791'
PANN_SHA = '0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31'
PROJECTION_SEED = 20261025

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def media_rel(sid):
    bits = str(sid).split('$_$')
    if len(bits)!=2 or any(not b or b in ('.','..') or '/' in b or '\\' in b for b in bits):
        raise ValueError(f'Invalid sample id: {sid}')
    return f'{bits[0]}/{bits[1]}.mp4'

def derangement(n, seed):
    if n<2: raise ValueError('Need two examples to shuffle')
    order = np.random.default_rng(seed).permutation(n)
    ix = np.empty(n, dtype=np.int64); ix[order] = np.roll(order, 1)
    assert np.all(ix!=np.arange(n))
    return ix

def scale_fit(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0); std = x.std(0)
    # Near-constant channels remain bounded; all statistics use training rows only.
    std = np.where(std<1e-6, 1., std)
    return mean, std

def scale_apply(x, stats):
    out = ((np.asarray(x, dtype=np.float64)-stats[0])/stats[1]).astype(np.float32)
    if not np.isfinite(out).all(): raise ValueError('Nonfinite normalized audio')
    return out

def representations(parts, embeds, out):
    dim = parts['train']['audio'].shape[1]
    original = scale_fit(parts['train']['audio'])
    pre = scale_fit(embeds['train'])
    # Fixed, label-free Gaussian map, NOT fitted on validation or sentiment labels.
    projection = (np.random.default_rng(PROJECTION_SEED).standard_normal((2048,dim))/np.sqrt(2048)).astype(np.float32)
    projected = {s:scale_apply(embeds[s],pre)@projection for s in SPLITS}
    post = scale_fit(projected['train'])
    audio = {s:scale_apply(projected[s],post) for s in SPLITS}
    permutations = {s:derangement(len(parts[s]['y']), PROJECTION_SEED+i) for i,s in enumerate(SPLITS)}
    atomic_npz(Path(out)/'representation_stats.npz', original_mean=original[0], original_std=original[1],
               embedding_mean=pre[0], embedding_std=pre[1], projection=projection,
               projected_mean=post[0], projected_std=post[1], **{'shuffle_'+s:permutations[s] for s in SPLITS})
    result = {}
    for group in GROUPS:
        result[group] = {}
        for s in SPLITS:
            a = parts[s]['audio'] if group=='paper_original' else scale_apply(parts[s]['audio'], original)
            if group=='panns_projected': a = audio[s]
            if group=='panns_shuffled': a = audio[s][permutations[s]]
            result[group][s] = dict(parts[s], audio=a)
    return result

def scores(y, pred, prob=None):
    from common import metric
    m = metric(y, pred); m['n'] = m.pop('n_test')
    if prob is not None:
        p = np.clip(prob,1e-7,1-1e-7)
        m['nll'] = float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
        m['brier'] = float(np.mean((prob-y)**2))
    return m

def verify_code():
    manifest = read_json(HERE/'bundle_manifest.json')
    for rel, digest in manifest['files'].items():
        if sha_file(HERE/rel)!=digest: raise RuntimeError(f'Code changed: {rel}')
    return sha_file(HERE/'bundle_manifest.json')

def record_json(path, value):
    # Existing artifacts are immutable: different inputs require a new output folder.
    path = Path(path)
    if path.exists() and read_json(path)!=value:
        raise RuntimeError(f'Configuration or inventory changed: {path}. Use a new output directory.')
    dump_json(path,value)
