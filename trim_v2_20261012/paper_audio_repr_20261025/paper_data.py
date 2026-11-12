"""Fix a common available-media subset; extract audio; never select using test data."""
import argparse, gc, time
from paper_common import *

def audit(root, raw_root, out, smoke=False):
    digest = sha_file(root/'MSA Datasets/MOSEI/Processed/aligned_50.pkl')
    if digest!=DATA_SHA: raise RuntimeError('Paper aligned MOSEI file differs from the archived input')
    raw = read_raw(root/'MSA Datasets','MOSEI')
    plan = dict(dataset='MOSEI', data_sha256=digest, pooling='historical_stored_frame_mean_including_padding',
                subset='available_media_non_neutral_train_valid', raw_root=str(raw_root.resolve()),
                smoke=smoke, test_evaluated=False, splits={}, availability={})
    excluded = []; all_ids = set()
    for s in SPLITS:
        # Do not prepare, index, select, or evaluate the test split.
        part = prepare(raw[s]); ids=part['ids']; labels=part['y']
        if all_ids.intersection(ids.tolist()): raise RuntimeError('Train/validation IDs overlap')
        all_ids.update(ids.tolist()); available=[]; missing=[]
        for i,sid in enumerate(ids):
            path = raw_root/media_rel(sid)
            if path.is_file() and path.stat().st_size>0: available.append(i)
            else:
                missing.append(i); excluded.append(dict(split=s,id=sid,label=int(labels[i]),relative_path=media_rel(sid),reason='missing_or_empty'))
        count=lambda ix: np.bincount(labels[ix],minlength=2).tolist()
        plan['availability'][s]=dict(total=len(ids),available=len(available),missing=len(missing),
                                     available_class_counts=count(available),missing_class_counts=count(missing))
        if min(count(available))<8: raise RuntimeError(f'Need at least 8 examples per class in {s}')
        ix=np.array(available,dtype=np.int64)
        if smoke:
            from model_tools import balanced_probe_indices
            ix=ix[balanced_probe_indices(labels[ix],42,n=16)]
        chosen={k:v[ix] for k,v in part.items()}
        atomic_npz(out/f'paper_{s}.npz',**chosen)
        plan['splits'][s]=[dict(id=str(ids[i]),label=int(labels[i]),relative_path=media_rel(ids[i]),
                               size_bytes=(raw_root/media_rel(ids[i])).stat().st_size) for i in ix]
    del raw;gc.collect()
    record_json(out/'inventory.json',plan)
    write_csv(out/'excluded_media.csv',excluded)
    print('PAPER_AUDIO_AUDIT',json.dumps(plan['availability']), 'smoke',smoke,'test_evaluated=False',flush=True)
    return plan

def preflight_bert(root, out):
    from bert_weight_check import compare_weights
    from bert_config_check import read_config, compare_config, CONFIG_METADATA
    bert=root/'MSA Datasets/pretrained_berts/bert_en'
    refs=read_json(HERE/'provenance/bert_reference.json')
    config_ref=next(r for r in refs if r['kind']=='bert_config')
    weights_ref=next(r for r in refs if r['kind']=='bert_weights')
    actual_sha=sha_file(bert/'config.json')
    config_check=dict(expected_file_sha256=config_ref['sha256'],actual_file_sha256=actual_sha,status='failed')
    try:
        actual=read_config(bert/'config.json')
        differences,expected_parameters,actual_parameters=compare_config(config_ref['config_reference'],actual)
        config_check.update(differences=differences,expected_parameters_sha256=expected_parameters,
                            actual_parameters_sha256=actual_parameters,
                            ignored_metadata_differences={k:dict(expected=config_ref['config_reference'].get(k),actual=actual.get(k))
                                for k in sorted(CONFIG_METADATA) if config_ref['config_reference'].get(k)!=actual.get(k)})
    except (ValueError,TypeError,KeyError) as exc:
        config_check['error']=str(exc);dump_json(out/'bert_config_check.json',config_check)
        raise RuntimeError(f'Invalid BERT configuration: {exc}. See {out / "bert_config_check.json"}') from exc
    if differences:
        dump_json(out/'bert_config_check.json',config_check)
        raise RuntimeError('BERT configuration parameters differ: '+json.dumps(differences,ensure_ascii=False)+
                           f'. See {out / "bert_config_check.json"}')
    config_check.update(status='passed',verification='exact_file_sha256' if actual_sha==config_ref['sha256'] else 'equivalent_bert_parameters')
    dump_json(out/'bert_config_check.json',config_check)
    print('PAPER_BERT_CONFIG_VERIFIED',config_check['verification'],'parameters_sha256',actual_parameters,flush=True)
    weight=next((bert/f for f in ('model.safetensors','pytorch_model.bin') if (bert/f).is_file()),None)
    if weight is None: raise FileNotFoundError('Missing local English BERT checkpoint')
    check=compare_weights(weight,weights_ref['tensor_reference'])
    if not check['passed']:
        dump_json(out/'bert_weight_check.json',check)
        raise RuntimeError(f'BERT tensors differ: {check}')
    check.update(weight=str(weight),weight_sha256=sha_file(weight),config_sha256=actual_sha,
                 config_verification=config_check)
    record_json(out/'bert_verified.json',check)
    print('PAPER_BERT_TENSORS_VERIFIED',str(check['matched_tensors'])+'/'+str(check['total_reference_tensors']),flush=True)
    return bert

def extract(plan,out,assets,cache,device):
    import torch
    import importlib.metadata
    from panns_frozen import load_encoder,decode_audio,extract_signal,CHECKPOINT_NAME
    checkpoint=assets/CHECKPOINT_NAME
    if sha_file(checkpoint)!=PANN_SHA: raise RuntimeError('PANNs checkpoint differs')
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    packages={name:importlib.metadata.version(name) for name in ('panns-inference','torchlibrosa','librosa','numpy','imageio-ffmpeg')}
    signature=dict(checkpoint_sha256=PANN_SHA,extractor_sha256=sha_file(HERE/'panns_frozen.py'),packages=packages,
                   torch=torch.__version__,device_type=torch.device(device).type,sample_rate=32000,
                   window_seconds=10,stride_seconds=5,padding='trailing_zeros',embedding_pool='window_mean',cudnn=False)
    record_json(cache/'signature.json',signature);record_json(out/'extraction_signature.json',signature)
    model=load_encoder(checkpoint,torch.device(device)); records=[];done=0
    total=sum(len(plan['splits'][s]) for s in SPLITS)
    for s in SPLITS:
        embeddings=[];ids=[]
        for row in plan['splits'][s]:
            path=Path(plan['raw_root'])/row['relative_path']
            if not path.is_file() or path.stat().st_size!=row['size_bytes']: raise RuntimeError(f'Media changed: {path}')
            digest=sha_file(path); key=hashlib.sha256(row['id'].encode()).hexdigest(); stored=cache/(key+'.npz')
            if stored.exists():
                item=load_npz(stored)
                if str(item['source_sha256'].item())!=digest: raise RuntimeError(f'Cached media changed: {path}')
            else:
                wave=decode_audio(path);events,embedding,windows=extract_signal(model,wave,torch.device(device))
                item=dict(events=events,embedding=embedding,seconds=np.array(len(wave)/32000),
                          windows=np.array(windows),rms=np.array(np.sqrt(np.mean(wave.astype(np.float64)**2))),source_sha256=np.array(digest))
                atomic_npz(stored,**item)
            embedding=item['embedding']
            if embedding.shape!=(2048,) or not np.isfinite(embedding).all(): raise RuntimeError(f'Invalid embedding: {path}')
            embeddings.append(embedding);ids.append(row['id']);done+=1
            records.append(dict(split=s,id=row['id'],relative_path=row['relative_path'],source_sha256=digest,
                                cache_sha256=sha_file(stored),seconds=float(item['seconds']),windows=int(item['windows']),rms=float(item['rms'])))
            if done==1 or done%25==0 or done==total:
                print('PAPER_AUDIO_EXTRACT',done,'/',total,s,row['id'],flush=True)
                dump_json(out/'extraction_progress.json',dict(done=done,total=total,status='extracting',test_evaluated=False))
        atomic_npz(out/f'embedding_{s}.npz',ids=np.array(ids),embedding=np.stack(embeddings))
    dump_json(out/'media_manifest.json',dict(records=records,test_evaluated=False))
    dump_json(out/'extraction_progress.json',dict(done=done,total=total,status='complete',test_evaluated=False))
    del model;gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def load_parts(out):
    plan=read_json(out/'inventory.json');parts={};embeds={}
    if set(plan['splits'])!=set(SPLITS): raise RuntimeError('Only train/valid allowed')
    for s in SPLITS:
        parts[s]=load_npz(out/f'paper_{s}.npz');e=load_npz(out/f'embedding_{s}.npz')
        ids=np.array([r['id'] for r in plan['splits'][s]])
        labels=np.array([r['label'] for r in plan['splits'][s]])
        if not np.array_equal(parts[s]['ids'],ids) or not np.array_equal(e['ids'],ids): raise RuntimeError('Embedding IDs are misaligned')
        if not np.array_equal(parts[s]['y'],labels): raise RuntimeError('Subset labels changed')
        if e['embedding'].shape!=(len(ids),2048): raise RuntimeError('Embedding dimensions differ')
        embeds[s]=e['embedding']
    return parts,embeds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE.parent.parent)
    ap.add_argument('--raw-root',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--assets',type=Path,default=HERE.parent/'audio_event_remote_20261024/assets')
    ap.add_argument('--cache',type=Path,default=HERE/'feature_cache');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--audit-only',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);verify_code()
    preflight_bert(a.root,a.out)
    plan=audit(a.root,a.raw_root or a.root/'MSA Datasets/MOSEI/Raw',a.out,a.smoke)
    if not a.audit_only: extract(plan,a.out,a.assets,a.cache,a.device)
if __name__=='__main__': main()
