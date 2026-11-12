"""Summarize all sealed runs; preserve every planned comparison and draw."""
import csv, shutil, zipfile
from collections import defaultdict
import numpy as np
from common import *
from interventions import paired_bootstrap

def condition_group(name):
    for prefix in ['cross_a_', 'cross_v_', 'cross_av_', 'within_av_']:
        if name.startswith(prefix): return prefix[:-1]
    return name

def summarize(out):
    out = Path(out); p = plan(); data = evaluation_data(out, 'IEMOCAP', 'test')
    seal = read(out / 'training_seal.json'); rows = []; predictions = {}; identities = {}; histories = {}
    for seed in p['seeds']:
        for variant in p['variants']:
            train = out / 'training/IEMOCAP' / variant / f'seed_{seed}'
            meta = read(train / 'complete.json')
            if meta['signature']['bundle_sha256'] != verify_code() or meta['signature']['protocol_sha256'] != sha(HERE / 'protocol.json'):
                raise RuntimeError('Training signature differs')
            if meta['test_evaluated']: raise RuntimeError('Training metadata contains test evaluation')
            if sha(train / 'best.pt') != seal['checkpoint_sha256'][(train / 'best.pt').relative_to(out).as_posix()]:
                raise RuntimeError('Sealed checkpoint changed')
            for name, h in meta['files'].items():
                if sha(train / name) != h: raise RuntimeError('Training artifact changed: ' + name)
            identities[(variant, seed)] = read(train / 'model_identity.json')
            with (train / 'history.csv').open(encoding='utf-8-sig', newline='') as f: histories[(variant, seed)] = list(csv.DictReader(f))
            folder = out / 'evaluation/IEMOCAP' / variant / f'seed_{seed}' / 'test'
            emeta = read(folder / 'complete.json')
            if emeta['signature']['training_seal_sha256'] != sha(out / 'training_seal.json') or emeta['selection_performed']:
                raise RuntimeError('Evaluation seal/selection status differs')
            for name, h in emeta['files'].items():
                if sha(folder / name) != h: raise RuntimeError('Evaluation artifact changed: ' + name)
            d = arrays(folder / 'predictions.npz')
            if not np.array_equal(d['ids'], data['ids']) or not np.array_equal(d['labels'], data['y']): raise RuntimeError('Prediction IDs/labels differ')
            predictions[(variant, seed)] = d
            with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f: rows.extend(csv.DictReader(f))
    for seed in p['seeds']:
        if len({identities[(v, seed)]['common_branch_initialization_sha256'] for v in p['variants'] if v != 'bert_mlp'}) != 1:
            raise RuntimeError('Common multimodal branch initialization differs')
        paired = {tuple((r['phase'], r['epoch'], r['modality_mask_sha256'], r['loader_rng_sha256']) for r in histories[(v, seed)]) for v in p['variants']}
        if len(paired) != 1: raise RuntimeError('Batch/mask streams differ across variants')
    review = out / 'review'; review.mkdir(exist_ok=True); csvout(review / 'all_condition_metrics.csv', rows)
    metrics_names = ['macro_f1', 'accuracy', 'weighted_f1', 'uar', 'nll', 'brier']
    per_seed = defaultdict(list)
    for row in rows: per_seed[(row['variant'], row['seed'], condition_group(row['condition']))].append(row)
    seed_rows = []
    for (variant, seed, group), rs in per_seed.items():
        seed_rows.append(dict(variant=variant, seed=int(seed), condition=group, draws=len(rs),
                              **{m: float(np.mean([float(r[m]) for r in rs])) for m in metrics_names}))
    csvout(review / 'seed_metrics.csv', seed_rows)
    grouped = defaultdict(list)
    for row in seed_rows: grouped[(row['variant'], row['condition'])].append(row)
    summary = []
    for (variant, condition), rs in grouped.items():
        if len(rs) != len(p['seeds']): raise RuntimeError('A planned seed is missing')
        item = dict(variant=variant, condition=condition, training_seeds=len(rs))
        for metric in metrics_names:
            x = [r[metric] for r in rs]; item[metric + '_mean'] = float(np.mean(x)); item[metric + '_sample_sd'] = float(np.std(x, ddof=1))
        summary.append(item)
    csvout(review / 'summary.csv', summary)
    full = {}; contrasts = []; counts = {}
    for variant in p['variants']:
        all_d = [predictions[(variant, s)] for s in p['seeds']]
        names = all_d[0]['condition_names'].tolist()
        if any(d['condition_names'].tolist() != names or not np.array_equal(d['eligible'], all_d[0]['eligible']) for d in all_d):
            raise RuntimeError('Intervention eligibility differs across seeds')
        pred = np.stack([d['logits'].argmax(-1) for d in all_d])
        full[variant] = pred[:, names.index('full')]
        condgroups = defaultdict(list)
        for k, name in enumerate(names): condgroups[condition_group(name)].append(k)
        for group, ix in condgroups.items():
            if group == 'full': continue
            keep = np.logical_and.reduce(all_d[0]['eligible'][ix])
            if not keep.any(): continue
            result = paired_bootstrap(full[variant], pred[:, ix], data['y'], data['source'], keep,
                                      p['bootstrap']['repeats'], p['bootstrap']['seed'])
            contrasts.append(dict(comparison='within_model', variant=variant, base='full', changed=group,
                                  eligible_n=int(keep.sum()), **result))
        counts[variant] = dict(parameters=[identities[(variant, s)]['parameters'] for s in p['seeds']],
                               non_bert_parameters=[identities[(variant, s)]['non_bert_parameters'] for s in p['seeds']],
                               fusion_budget=identities[(variant, p['seeds'][0])]['fusion_budget'])
    for baseline in p['variants']:
        if baseline == 'trim': continue
        result = paired_bootstrap(full[baseline], full['trim'], data['y'], data['source'], np.ones(len(data['y']), bool),
                                  p['bootstrap']['repeats'], p['bootstrap']['seed'])
        contrasts.append(dict(comparison='model', variant='trim', base=baseline, changed='trim', eligible_n=len(data['y']), **result))
    dump(review / 'paired_contrasts.json', contrasts); dump(review / 'parameter_counts.json', counts)
    audit = dict(new_training_jobs=len(predictions), expected_training_jobs=len(p['variants']) * len(p['seeds']),
                 parameter_and_stochastic_pairing_verified=True, all_checkpoints_sealed_before_test=True,
                 formal_test_evaluation_completed=True, fresh_untouched_external_validation=False,
                 selection='Session 4 Macro-F1 only', test_session='Ses05',
                 confidence_interval_scope=p['bootstrap']['scope'], protocol_sha256=sha(HERE / 'protocol.json'),
                 bundle_sha256=verify_code())
    dump(review / 'review_audit.json', audit)
    lines = ['# IEMOCAP six-class supplement', '', 'Fixed sessions: fit 1–3; selection 4; test 5. Five training seeds. No test-based selection.', '',
             '| System | Macro-F1 (%) | Accuracy (%) | Weighted F1 (%) | UAR (%) |', '| --- | ---: | ---: | ---: | ---: |']
    for variant in p['variants']:
        r = next(r for r in summary if r['variant'] == variant and r['condition'] == 'full')
        fmt = lambda m: f"{100*r[m+'_mean']:.2f} ± {100*r[m+'_sample_sd']:.2f}"
        lines.append('| ' + variant + ' | ' + ' | '.join(fmt(m) for m in ['macro_f1', 'accuracy', 'weighted_f1', 'uar']) + ' |')
    lines.extend(['', 'Values are means and sample SD across training seeds. All perturbation draws are nested within each seed.', '',
                  'Paired intervals are pointwise and conditional on Session 5 speakers. This is a fixed-session extension, not five-fold SOTA evaluation or an untouched-corpus claim.', '',
                  'Raw transcripts were re-tokenized with the pinned English BERT vocabulary; supplied RoBERTa features were unused. AV extractor training provenance remains unverified.'])
    (review / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    archive = out / 'kbs_iemocap_review_20261111.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if not path.is_file() or path == archive: continue
            rel = path.relative_to(out)
            if rel.parts[0] == 'data':
                if path.name != 'prepared.json': continue
            elif path.suffix not in ['.json', '.csv', '.md', '.npz']: continue
            z.write(path, rel.as_posix())
        for name in ['protocol.json', 'PROTOCOL.md', 'bundle_manifest.json']:
            z.write(HERE / name, 'source/' + name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError('Corrupt review archive')
    print('IEMO_REVIEW_COMPLETE', archive, flush=True)
    return archive
