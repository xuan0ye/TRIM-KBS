"""Session-disjoint input preparation; supplied RoBERTa arrays are not used."""
import pickle
from pathlib import Path
import numpy as np
from common import *

class FeatureReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {'numpy.core.multiarray', 'numpy.core._multiarray_umath', 'numpy._core.multiarray'} and name in {'_reconstruct', 'scalar'}:
            return getattr(np.core.multiarray, name)
        if (module, name) in {('numpy', 'ndarray'), ('numpy', 'dtype'), ('_codecs', 'encode'), ('builtins', 'set')}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')

def bert_path(root): return Path(root) / 'MSA Datasets/pretrained_berts/bert_en'

def verify_bert(root, out):
    from transformers import BertConfig
    from bert_weight_check import compare_weights
    directory = bert_path(root)
    reference = read(HERE / 'provenance/data_reference.json')
    if sha(directory / 'vocab.txt') != reference['vocab_sha256']: raise RuntimeError('BERT vocabulary differs')
    config = BertConfig.from_pretrained(str(directory), local_files_only=True).to_dict()
    wanted = read(HERE / 'provenance/bert_architecture_reference.json')['bert_en']
    diff = {k: [v, config.get(k)] for k, v in wanted.items() if config.get(k) != v}
    if diff: raise RuntimeError('BERT architecture differs: ' + str(diff))
    refs = read(HERE / 'provenance/input_reference.json')['inputs']
    ref = next(x for x in refs if x.get('kind') == 'bert_weights')
    weights = directory / 'model.safetensors'
    if not weights.exists(): weights = directory / 'pytorch_model.bin'
    check = compare_weights(weights, ref['tensor_reference'])
    if not check['passed']: raise RuntimeError('BERT tensor identity differs')
    record = dict(weights_sha256=sha(weights), vocab_sha256=sha(directory / 'vocab.txt'),
                  config_sha256=sha(directory / 'config.json'), tensor_check=check)
    dump(Path(out) / 'bert_identity.json', record)
    print('IEMO_BERT_VERIFIED', check['matched_tensors'], flush=True)
    return record

def flatten(raw, keys, tokenizer, max_tokens):
    keys = sorted(keys); ids = []; source = []; y = []; texts = []; aa = []; vv = []
    for key in keys:
        n = len(raw[0][key])
        if any(len(raw[k][key]) != n for k in [1, 2, 7, 8, 9]): raise ValueError('Utterance alignment differs: ' + key)
        for j in range(n):
            sid = str(raw[0][key][j])
            if not sid.startswith(key + '_'): raise ValueError('Unexpected utterance ID: ' + sid)
            ids.append(sid); source.append(key); y.append(int(raw[2][key][j]))
            text = raw[9][key][j]
            if not isinstance(text, str): raise TypeError('Transcript must be a string')
            texts.append(text.strip())
            aa.append(np.asarray(raw[7][key][j], np.float32)); vv.append(np.asarray(raw[8][key][j], np.float32))
    if len(set(ids)) != len(ids): raise ValueError('Repeated utterance IDs')
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=max_tokens, return_token_type_ids=True)
    text = np.stack([tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids']], axis=1).astype(np.int64)
    a, v = np.stack(aa), np.stack(vv)
    if a.shape[1] != 1582 or v.shape[1] != 342 or not np.isfinite(a).all() or not np.isfinite(v).all():
        raise ValueError('Unexpected IEMOCAP AV dimensions/values')
    labels = np.asarray(y, np.int8)
    if not np.array_equal(np.unique(labels), np.arange(6)): raise ValueError('All six classes required in fixed split')
    return dict(ids=np.asarray(ids), source=np.asarray(source), y=labels, text=text, audio=a, vision=v,
                observed=np.stack([np.any(a != 0, 1), np.any(v != 0, 1)], 1)), sum(not s for s in texts)

def fit_scaler(fit):
    result = {}
    for j, name in enumerate(['audio', 'vision']):
        x = fit[name][fit['observed'][:, j]].astype(np.float64)
        if not len(x): raise ValueError('No observed training ' + name)
        result[name + '_mean'] = x.mean(0)
        result[name + '_scale'] = np.maximum(x.std(0), 1e-6)
    return result

def apply_scaler(data, scaler):
    for j, name in enumerate(['audio', 'vision']):
        mask = data['observed'][:, j]
        x = np.zeros_like(data[name], dtype=np.float32)
        x[mask] = ((data[name][mask].astype(np.float64) - scaler[name + '_mean']) / scaler[name + '_scale']).astype(np.float32)
        if not np.isfinite(x).all(): raise ValueError('Scaling produced nonfinite input')
        data[name] = x
    return data

def prepare(root, out):
    from transformers import BertTokenizer
    p = plan(); ref = read(HERE / 'provenance/data_reference.json'); path = Path(root) / ref['relative_path']
    digest = sha(path)
    if digest != ref['sha256']: raise RuntimeError('Remote IEMOCAP differs from audited local file: ' + str(path))
    directory = bert_path(root)
    if sha(directory / 'vocab.txt') != ref['vocab_sha256']: raise RuntimeError('Vocabulary differs')
    sig = dict(data_sha256=digest, vocab_sha256=ref['vocab_sha256'], protocol_sha256=sha(HERE / 'protocol.json'), bundle_sha256=verify_code())
    target = Path(out) / 'data/IEMOCAP'; meta = target / 'prepared.json'
    if meta.exists():
        old = read(meta)
        if old['signature'] != sig: raise RuntimeError('Prepared cache differs; use a new run directory')
        for name, h in old['files'].items():
            if sha(target / name) != h: raise RuntimeError('Prepared file changed: ' + name)
        return old
    with path.open('rb') as f: raw = FeatureReader(f, encoding='latin1').load()
    if len(raw) != 12 or any(set(raw[i]) != set(raw[0]) for i in range(1, 10)): raise ValueError('Expected 12-field IEMOCAP schema')
    train, test = set(raw[10]), set(raw[11])
    if train & test or train | test != set(raw[0]): raise ValueError('Invalid file train/test partition')
    fit = {s for s in train if s[:5] in p['fit_sessions']}
    dev = {s for s in train if s[:5] in p['dev_sessions']}
    if fit & dev or fit | dev != train or any(s[:5] not in p['test_sessions'] for s in test): raise ValueError('Unexpected sessions')
    tokenizer = BertTokenizer(vocab_file=str(directory / 'vocab.txt'), do_lower_case=True, model_max_length=p['max_tokens'])
    datasets = {}; coverage = {}
    for role, keys in [('fit', fit), ('dev', dev), ('test', test)]:
        d, empty = flatten(raw, keys, tokenizer, p['max_tokens']); datasets[role] = d
        coverage[role] = dict(n=len(d['y']), dialogues=len(keys), sessions=sorted({s[:5] for s in keys}),
                              empty_transcripts=empty, native_absent_audio=int((~d['observed'][:, 0]).sum()), native_absent_visual=int((~d['observed'][:, 1]).sum()))
    if any(set(datasets[x]['ids']) & set(datasets[y]['ids']) for x, y in [('fit', 'dev'), ('fit', 'test'), ('dev', 'test')]): raise ValueError('Split overlap')
    scaler = fit_scaler(datasets['fit']); npz(target / 'av_scaler.npz', **scaler)
    for role, d in datasets.items(): npz(target / f'{role}.npz', **apply_scaler(d, scaler))
    record = dict(signature=sig, coverage=coverage, files={x.name: sha(x) for x in target.glob('*.npz')},
                  supplied_text_features_unused=True, test_used_for_training_or_selection=False,
                  test_arrays_prepared_for_integrity_only=True, historical_corpus=True)
    dump(meta, record); dump(Path(out) / 'data_inventory.json', record)
    print('IEMO_DATA_PREPARED', coverage, flush=True)
    return record
