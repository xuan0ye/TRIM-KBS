"""Post-seal Session 5 evaluation. Cached branches preserve native absence masks."""
import gc, time
import numpy as np
import torch
from common import *
from data import bert_path
from models import make_model
from interventions import *

def encode_bank(model, data, device, batch=32):
    model.eval(); ts = []; aa = []; vv = []
    with torch.inference_mode():
        for start in range(0, len(data['y']), batch):
            ix = slice(start, start + batch)
            t = torch.as_tensor(data['text'][ix], device=device)
            a = torch.as_tensor(data['audio'][ix], device=device); v = torch.as_tensor(data['vision'][ix], device=device)
            mask = torch.as_tensor(data['observed'][ix], device=device)
            ht = model.encode_text(t); ha, hv = model.encode_av(a, v, mask)
            ts.append(ht.cpu()); aa.append(ha.cpu()); vv.append(hv.cpu())
        ma, mv = model.absent_states(1, device)
        za, zv = model.encode_av(torch.zeros((1, data['audio'].shape[1]), device=device), torch.zeros((1, data['vision'].shape[1]), device=device))
    return dict(t=torch.cat(ts), a=torch.cat(aa), v=torch.cat(vv), ma=ma.cpu(), mv=mv.cpu(), za=za.cpu(), zv=zv.cpu())

def predictor(model, bank, device):
    def predict(ti, ai, vi=None, mode='full'):
        ti = np.asarray(ti); ai = np.asarray(ai); vi = ai if vi is None else np.asarray(vi)
        t = bank['t'][ti].to(device)
        a = bank['a'][ai] if mode in ['full', 'text_audio'] else bank['ma'].expand(len(ti), -1)
        v = bank['v'][vi] if mode in ['full', 'text_visual'] else bank['mv'].expand(len(ti), -1)
        if mode == 'zero_av_present': a = bank['za'].expand(len(ti), -1); v = bank['zv'].expand(len(ti), -1)
        with torch.inference_mode(): z = model.fuse(t, a.to(device), v.to(device)).float().cpu().numpy()
        return z - z.mean(-1, keepdims=True)
    return predict

def infer(fn, ti, ai, vi=None, mode='full', batch=128):
    vi = ai if vi is None else vi
    return np.concatenate([fn(ti[s:s+batch], ai[s:s+batch], vi[s:s+batch], mode) for s in range(0, len(ti), batch)])

def evaluate_one(root, out, dataset, variant, seed, role, device):
    out = Path(out); p = plan(); data = evaluation_data(out, dataset, role)
    maps = build_maps(data['source'], p['interventions']['repeats'], p['interventions']['seed'])
    map_path = out / 'interventions/maps.npz'
    if map_path.exists():
        old = arrays(map_path)
        if set(old) != set(maps) or any(not np.array_equal(old[k], v) for k, v in maps.items()): raise RuntimeError('Donor maps changed')
    else: npz(map_path, **maps)
    seal_path = out / 'training_seal.json'; seal = read(seal_path)
    folder0 = out / 'training' / dataset / variant / f'seed_{seed}'; weights = folder0 / 'best.pt'
    want = seal['checkpoint_sha256'][weights.relative_to(out).as_posix()]
    if sha(weights) != want: raise RuntimeError('Sealed checkpoint changed')
    folder = out / 'evaluation' / dataset / variant / f'seed_{seed}' / role; folder.mkdir(parents=True, exist_ok=True)
    sig = dict(checkpoint_sha256=want, training_seal_sha256=sha(seal_path), maps_sha256=sha(map_path),
               data_sha256=sha(out / 'data' / dataset / 'test.npz'), protocol_sha256=sha(HERE / 'protocol.json'), bundle_sha256=verify_code())
    if (folder / 'complete.json').exists():
        old = read(folder / 'complete.json')
        if old['signature'] != sig: raise RuntimeError('Evaluation cache changed')
        for name, h in old['files'].items():
            if sha(folder / name) != h: raise RuntimeError('Evaluation file changed: ' + name)
        print('IEMO_EVALUATION_CACHED', variant, seed, flush=True); return
    runtime(device, 20261111)
    model = make_model(bert_path(root), data['audio'].shape[1], data['vision'].shape[1], variant, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True); model.eval()
    tick = time.monotonic(); bank = encode_bank(model, data, device); fn = predictor(model, bank, device)
    ix = np.arange(len(data['y'])); q = ix[:min(8, len(ix))]
    with torch.inference_mode():
        z, _ = model(torch.as_tensor(data['text'][q], device=device), torch.as_tensor(data['audio'][q], device=device),
                     torch.as_tensor(data['vision'][q], device=device), torch.as_tensor(data['observed'][q], device=device))
        z = z.cpu().numpy(); z -= z.mean(1, keepdims=True)
    error = float(np.abs(z - fn(q, q)).max())
    if error > 1e-4: raise RuntimeError('Cached branch/direct forward mismatch: ' + str(error))
    names = []; scores = []; eligible = []
    def add(name, value, keep=None):
        names.append(name); scores.append(value); eligible.append(np.ones(len(ix), bool) if keep is None else keep)
    for mode in ['full', 'text_only', 'text_audio', 'text_visual', 'zero_av_present']:
        add(mode, infer(fn, ix, ix, mode=mode))
    for rep in range(p['interventions']['repeats']):
        for prefix, key, which in [('cross_a', 'cross', 'a'), ('cross_v', 'cross', 'v'), ('cross_av', 'cross', 'av'), ('within_av', 'within', 'av')]:
            donor = maps[f'{key}_{rep:02}']; keep = donor >= 0
            value = np.full((len(ix), 6), np.nan, np.float32)
            if keep.any():
                ai = donor[keep] if which in ['a', 'av'] else ix[keep]
                vi = donor[keep] if which in ['v', 'av'] else ix[keep]
                value[keep] = infer(fn, ix[keep], ai, vi)
            add(f'{prefix}_{rep:02}', value, keep)
    banks = []; halves = []; details = []
    for number, anchor in enumerate(anchors_for(len(ix), p['emap']['anchors'], p['emap']['banks'], p['emap']['seed'])):
        value, half, meta = anchor_emap(len(ix), anchor, lambda a, b: fn(a, b))
        banks.append(value); halves.append(half); details.append(dict(bank=number, **meta))
        print('IEMO_EMAP_BANK', variant, seed, number + 1, flush=True)
    projected = np.mean(banks, axis=0)
    emapmeta = dict(approximation='two independent 64-anchor centered-logit projections', banks=details,
                    bank_centered_logit_rmse=float(np.sqrt(np.square(banks[0] - banks[1]).mean())))
    if variant in ['bert_mlp', 'additive']:
        err = float(np.abs(projected - scores[0]).max()); emapmeta['additive_max_centered_logit_error'] = err
        if err > 1e-4: raise RuntimeError('Projection changed an additive classifier: ' + str(err))
    add('emap', projected)
    for i, (bank_score, half) in enumerate(zip(banks, halves)):
        add(f'emap_bank_{i}', bank_score); add(f'emap_half_{i}', half)
    # Score exactly the precision retained in the returned prediction artifact.
    logits = np.stack(scores).astype(np.float32); masks = np.stack(eligible); rows = []; base = logits[0].argmax(-1)
    for k, name in enumerate(names):
        keep = masks[k]
        if not keep.any(): continue
        changed = logits[k, keep].argmax(-1); y = data['y'][keep]
        rows.append(dict(dataset=dataset, variant=variant, seed=seed, split=role, condition=name,
                         eligible_fraction=float(keep.mean()), **metrics(y, logits[k, keep]),
                         changed_fraction=float((base[keep] != changed).mean()),
                         corrected=int(((base[keep] != y) & (changed == y)).sum()),
                         broken=int(((base[keep] == y) & (changed != y)).sum())))
    npz(folder / 'predictions.npz', ids=data['ids'], labels=data['y'], source=data['source'], condition_names=np.asarray(names),
        logits=logits.astype(np.float32), eligible=masks)
    csvout(folder / 'metrics.csv', rows); dump(folder / 'emap.json', emapmeta)
    files = ['predictions.npz', 'metrics.csv', 'emap.json']
    dump(folder / 'complete.json', dict(signature=sig, files={name: sha(folder / name) for name in files},
                                      cached_branch_max_centered_logit_error=error, seconds=time.monotonic() - tick,
                                      test_evaluated=True, selection_performed=False, fixed_session_holdout=True))
    print('IEMO_EVALUATION_DONE', variant, seed, 'conditions', len(names), flush=True)
    del model, bank; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
