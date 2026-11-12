"""Verify, prepare, smoke, train 25, seal all, then evaluate Session 5."""
import argparse, contextlib, gc, os, shutil, sys, time
import numpy as np
from common import *

def dependencies(device):
    import torch, transformers
    if not torch.__version__.startswith('2.4.'): raise RuntimeError('Expected PyTorch 2.4.x; got ' + torch.__version__)
    if transformers.__version__ != '4.49.0': raise RuntimeError('Expected Transformers 4.49.0')
    # The training loader consumes NumPy arrays and evaluation exports tensors.
    try:
        values = np.array([1.0, 2.0], dtype=np.float32)
        result = torch.from_numpy(values).numpy()
    except (RuntimeError, TypeError, ImportError) as exc:
        raise RuntimeError(
            f'PyTorch {torch.__version__} cannot exchange arrays with NumPy '
            f'{np.__version__}: {exc}'
        ) from exc
    if result.dtype != values.dtype or not np.array_equal(result, values):
        raise RuntimeError(f'Unexpected NumPy {np.__version__} tensor conversion result')
    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; activate wjhenv.')
    return dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, device=device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

@contextlib.contextmanager
def global_lock():
    path = HERE.parent / 'logs/suite.lock'; path.parent.mkdir(exist_ok=True)
    with path.open('a+') as handle:
        if os.name != 'nt':
            import fcntl
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('IEMO_WAITING_FOR_SHARED_SUITE_LOCK; existing experiment continues', flush=True)
                fcntl.flock(handle, fcntl.LOCK_EX)
        yield

def preflight(root, out, device, quick=False):
    code = verify_code(); rt = dependencies(device)
    ref = read(HERE / 'provenance/data_reference.json')
    for path in [Path(root) / ref['relative_path'], Path(root) / 'MSA Datasets/pretrained_berts/bert_en/config.json', Path(root) / 'MSA Datasets/pretrained_berts/bert_en/vocab.txt']:
        if not path.is_file(): raise FileNotFoundError(path)
    if quick:
        print('IEMO_QUICK_CHECK_OK', rt, flush=True); return
    from data import verify_bert
    verify_bert(root, out)
    dump(Path(out) / 'runtime.json', dict(runtime=rt, bundle_sha256=code, protocol_sha256=sha(HERE / 'protocol.json')))

def disk_check(out):
    p = plan(); jobs = len(p['variants']) * len(p['seeds']); required = jobs * 125_000_000 * 4 + 5_000_000_000
    free = shutil.disk_usage(out).free
    if free < required: raise RuntimeError(f'Need {required/1e9:.1f} GB free for checkpoints; available {free/1e9:.1f} GB')
    print('IEMO_DISK_OK', 'reserved_GB', required / 1e9, 'free_GB', round(free / 1e9, 1), flush=True)

def smoke(root, out, device):
    import torch
    from data import bert_path
    from models import make_model, configure_phase, optimizer
    from train import loss_function
    p = plan(); sig = dict(bundle_sha256=verify_code(), protocol_sha256=sha(HERE / 'protocol.json'), bert_identity_sha256=sha(Path(out) / 'bert_identity.json'))
    path = Path(out) / 'smoke_check.json'
    if path.exists() and read(path)['signature'] == sig:
        print('IEMO_SMOKE_CACHED', flush=True); return
    tr = selection_data(out, 'IEMOCAP', 'fit'); ix = np.concatenate([np.flatnonzero(tr['y'] == y)[:2] for y in range(6)])
    t = torch.as_tensor(tr['text'][ix], device=device); a = torch.as_tensor(tr['audio'][ix], device=device)
    v = torch.as_tensor(tr['vision'][ix], device=device); y = torch.as_tensor(tr['y'][ix], dtype=torch.long, device=device)
    mask = torch.as_tensor(tr['observed'][ix], device=device); records = []
    for variant in p['variants']:
        runtime(device, 101); model = make_model(bert_path(root), a.shape[1], v.shape[1], variant).to(device)
        configure_phase(model, 0); opt = optimizer(model, 0); model.train()
        z, aux = model(t, a, v, mask, auxiliary=True)
        loss = loss_function(tr['y'], device)(z, y) + .02 * aux['nce'] + .02 * aux['ot']
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all(): raise RuntimeError('Nonfinite gradient: ' + name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); model.eval()
        with torch.inference_mode():
            off = torch.zeros_like(mask); x, _ = model(t, a, v, off); xx, _ = model(t, a * 23 + 1, v * 17 - 8, off)
            error = float((x - xx).abs().max())
        if error > 1e-5: raise RuntimeError('Missing-input invariance failed')
        records.append(dict(variant=variant, finite_training_step=True, absent_input_max_error=error, identity=model.identity()))
        print('IEMO_SMOKE_OK', variant, flush=True); del model, opt; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    dump(path, dict(signature=sig, status='passed', records=records, formal_training_jobs=0, test_evaluated=False))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['check', 'prepare', 'smoke', 'train', 'evaluate', 'summarize', 'run'])
    ap.add_argument('--root', type=Path, default=HERE.parent.parent); ap.add_argument('--out', type=Path, default=HERE / 'runs/extension')
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--quick', action='store_true'); args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'check': preflight(args.root, args.out, args.device, args.quick); return
    with global_lock():
        if args.mode in ['prepare', 'smoke', 'train', 'run']:
            preflight(args.root, args.out, args.device)
            from data import prepare
            prepare(args.root, args.out)
            if args.mode == 'prepare': return
            smoke(args.root, args.out, args.device)
            if args.mode == 'smoke': return
        if args.mode in ['train', 'run']:
            from train import train_one, seal_training
            disk_check(args.out); p = plan(); total = len(p['variants']) * len(p['seeds']); done = 0
            for seed in p['seeds']:
                for variant in p['variants']:
                    train_one(args.root, args.out, 'IEMOCAP', variant, seed, args.device); done += 1
                    dump(args.out / 'suite_progress.json', dict(stage='training', done=done, total=total, test_evaluated=False))
            seal_training(args.out)
            if args.mode == 'train': return
        if args.mode in ['evaluate', 'run']:
            from train import seal_training
            from evaluate import evaluate_one
            seal_training(args.out); p = plan(); done = 0; total = len(p['variants']) * len(p['seeds'])
            for seed in p['seeds']:
                for variant in p['variants']:
                    evaluate_one(args.root, args.out, 'IEMOCAP', variant, seed, 'test', args.device); done += 1
                    dump(args.out / 'suite_progress.json', dict(stage='post_seal_test_evaluation', done=done, total=total, selection_performed=False))
        if args.mode in ['summarize', 'evaluate', 'run']:
            from summarize import summarize
            archive = summarize(args.out)
            dump(args.out / 'suite_progress.json', dict(stage='complete', review_zip=str(archive), selection_performed=False))

if __name__ == '__main__': main()
