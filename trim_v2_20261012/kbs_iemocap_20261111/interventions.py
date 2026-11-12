"""Label-independent donors and vector-logit empirical additive projection."""
import numpy as np

def cross_source(source, seed):
    source = np.asarray(source); names, counts = np.unique(source, return_counts=True); n = len(source)
    if n < 2 or counts.max() > n // 2: return np.full(n, -1, np.int64)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(np.flatnonzero(source == v)) for v in rng.permutation(names)])
    donor = np.empty(n, np.int64); donor[order] = np.roll(order, int(rng.integers(counts.max(), n - counts.max() + 1)))
    assert len(np.unique(donor)) == n and np.all(source[donor] != source)
    return donor

def within_source(source, seed):
    rng = np.random.default_rng(seed); out = np.full(len(source), -1, np.int64)
    for name in np.unique(source):
        ix = np.flatnonzero(source == name)
        if len(ix) < 2: continue
        order = rng.permutation(ix); out[order] = np.roll(order, int(rng.integers(1, len(ix))))
    keep = out >= 0
    assert np.all(source[out[keep]] == source[keep]) and np.all(out[keep] != np.flatnonzero(keep))
    return out

def build_maps(source, repeats, seed):
    maps = {}
    for rep in range(repeats):
        maps[f'cross_{rep:02}'] = cross_source(source, seed + 101 * rep)
        maps[f'within_{rep:02}'] = within_source(source, seed + 101 * rep + 1)
    if any(np.any(x < 0) for k, x in maps.items() if k.startswith('cross')):
        raise ValueError('No complete cross-dialogue bijection exists')
    return maps

def anchors_for(n, count, banks, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(n, min(count, n), replace=False) for _ in range(banks)]

def anchor_emap(n, anchors, predict, pair_batch=128):
    anchors = np.asarray(anchors); m = len(anchors)
    r = np.empty((n, m, 6), np.float64); c = np.empty_like(r)
    for start in range(0, n * m, pair_batch):
        flat = np.arange(start, min(start + pair_batch, n * m)); i, j = flat // m, flat % m
        r[i, j] = predict(i, anchors[j]); c[i, j] = predict(anchors[j], i)
    grid = np.empty((m, m, 6), np.float64)
    for start in range(0, m * m, pair_batch):
        flat = np.arange(start, min(start + pair_batch, m * m)); i, j = flat // m, flat % m
        grid[i, j] = predict(anchors[i], anchors[j])
    full = r.mean(1) + c.mean(1) - grid.mean((0, 1)); h = max(1, m // 2)
    half = r[:, :h].mean(1) + c[:, :h].mean(1) - grid[:h, :h].mean((0, 1))
    if not np.isfinite(full).all(): raise ValueError('Nonfinite EMAP projection')
    return full, half, dict(anchors=m, pair_evaluations=2 * n * m + m * m,
                            half_to_full_centered_logit_rmse=float(np.sqrt(np.square(full - half).mean())))

def f1_counts(c):
    c = np.asarray(c, dtype=np.float64)
    diag = np.diagonal(c, axis1=-2, axis2=-1)
    return (2 * diag / np.maximum(1, c.sum(-1) + c.sum(-2))).mean(-1)

def clustered_counts(y, pred, source, eligible):
    out = []
    for name in np.unique(source):
        keep = (source == name) & eligible
        out.append(np.bincount(6 * y[keep].astype(int) + pred[keep].astype(int), minlength=36).reshape(6, 6))
    return np.asarray(out, np.float64)

def paired_bootstrap(base, changed, y, source, eligible, repetitions=2000, seed=20261113):
    # Class decisions: seed x draw x utterance. Draws stay nested within seed.
    base = np.asarray(base); changed = np.asarray(changed)
    if base.ndim == 2: base = base[:, None, :]
    if changed.ndim == 2: changed = changed[:, None, :]
    s, r, n = changed.shape
    if base.shape[0] != s or base.shape[-1] != n: raise ValueError('Paired prediction shape differs')
    v = len(np.unique(source))
    cb = np.stack([clustered_counts(y, b[0], source, eligible) for b in base])
    cc = np.stack([[clustered_counts(y, x, source, eligible) for x in z] for z in changed])
    point = 100 * (f1_counts(cc.sum(2)).mean(1) - f1_counts(cb.sum(1))).mean()
    rng = np.random.default_rng(seed); conditional = []; crossed = []
    for start in range(0, repetitions, 32):
        b = min(32, repetitions - start); w = rng.multinomial(v, np.full(v, 1 / v), size=b)
        mb = f1_counts(np.einsum('svij,bv->bsij', cb, w))
        mc = f1_counts(np.einsum('srvij,bv->bsrij', cc, w)).mean(-1)
        delta = mc - mb; conditional.extend(100 * delta.mean(1))
        ix = rng.integers(0, s, size=(b, s)); crossed.extend(100 * np.take_along_axis(delta, ix, 1).mean(1))
    return dict(delta_macro_f1_pp=float(point), dialogue_ci95=list(map(float, np.quantile(conditional, [.025, .975]))),
                dialogue_and_seed_ci95=list(map(float, np.quantile(crossed, [.025, .975]))),
                bootstrap_repetitions=repetitions, training_seeds=s, test_dialogues=v,
                scope='pointwise, conditional on fixed Session 5 speakers; not a population speaker interval')
