"""Shared validation-only audio interventions and bounded event residual pilot."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('PAPER_PARENT_DIR',HERE.parent/'paper_audio_repr_20261025')).resolve()
sys.path.insert(0,str(PARENT))
import paper_common as pc
import torch
from model_tools import encode,fuse,runtime
from paper_train import make_model

EVENTS=('Music','Background music','Applause','Explosion','Gunshot, gunfire','Fireworks',
        'Siren','Dog','Vehicle','Traffic noise, roadway noise','Door','Doorbell','Glass',
        'Wind','Rain','Thunder','Thunderstorm')
MODES=('T','TA','TV','TAV')
BITS={'T':(1,0,0),'TA':(1,1,0),'TV':(1,0,1),'TAV':(1,1,1)}
LABEL_SHA='cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429'

def verify_release():
    manifest=pc.read_json(HERE/'bundle_manifest.json')
    for rel,digest in manifest['files'].items():
        if pc.sha_file(HERE/rel)!=digest:raise RuntimeError(f'Diagnostic code changed: {rel}')
    pc.verify_code()
    if pc.sha_file(PARENT/'bundle_manifest.json')!=manifest['parent_manifest_sha256']:
        raise RuntimeError('Expected the completed config-fix parent release.')
    return pc.sha_file(HERE/'bundle_manifest.json')

def cross_video_permutation(ids,seed):
    """A bijection with no fixed clip AND no same-video donor; no labels used."""
    ids=np.asarray(ids);videos=np.array([str(x).split('$_$')[0] for x in ids])
    names,counts=np.unique(videos,return_counts=True);n=len(ids);maximum=int(counts.max())
    if maximum>n//2:raise ValueError('Cross-video bijection impossible: one video supplies over half the samples.')
    rng=np.random.default_rng(seed)
    order=np.concatenate([rng.permutation(np.flatnonzero(videos==v)) for v in rng.permutation(names)])
    offset=int(rng.integers(maximum,n-maximum+1))
    ix=np.empty(n,dtype=np.int64);ix[order]=np.roll(order,offset)
    if not np.all(videos[ix]!=videos) or len(np.unique(ix))!=n:raise AssertionError('Invalid cross-video permutation')
    return ix

def build_audio(parts,embeds,stats,group):
    result={}
    for s in pc.SPLITS:
        a=parts[s]['audio']
        if group=='paper_normalized':a=pc.scale_apply(a,(stats['original_mean'],stats['original_std']))
        if group in ('panns_projected','panns_shuffled'):
            a=pc.scale_apply(embeds[s],(stats['embedding_mean'],stats['embedding_std']))@stats['projection']
            a=pc.scale_apply(a,(stats['projected_mean'],stats['projected_std']))
            if group=='panns_shuffled':a=a[stats['shuffle_'+s]]
        result[s]=dict(parts[s],audio=a)
    return result

def feature_signature(run,group,split):
    complete=pc.read_json(run/group/'seed_42/complete.json')
    return dict(checkpoint_sha256=complete['artifacts']['best.pt'],inputs=complete['signature']['inputs'],
                group=group,split=split,torch=torch.__version__,cache_version=1)

def verify_parent(run):
    inputs=None;identity=None
    for group in pc.GROUPS:
        done=pc.read_json(run/group/'seed_42/complete.json')
        if done['status']!='complete' or done['test_evaluated']:raise RuntimeError('Parent not completed validation-only.')
        if done['signature']['seed']!=42:raise RuntimeError('Expected completed seed 42 pilot')
        if inputs is None:inputs=done['signature']['inputs'];identity=done['identity']
        if inputs!=done['signature']['inputs'] or identity!=done['identity']:raise RuntimeError('Parent group inputs differ')
        for rel,digest in done['artifacts'].items():
            if pc.sha_file(run/group/'seed_42'/rel)!=digest:raise RuntimeError(f'Parent artifact changed: {group}/{rel}')
    for rel,digest in inputs.items():
        path=PARENT/'bundle_manifest.json' if rel=='bundle_manifest_sha256' else run/rel
        if pc.sha_file(path)!=digest:raise RuntimeError(f'Parent input changed: {path}')
    import transformers
    sig=pc.read_json(run/'paper_original/seed_42/complete.json')['signature']
    if (torch.__version__,transformers.__version__,np.__version__)!=(sig['torch'],sig['transformers'],sig['numpy']):
        raise RuntimeError('Use the same torch/transformers/numpy environment as the parent pilot.')
    return inputs

def load_backbone(root,run,group,a_dim,v_dim,device):
    from bert_config_check import read_config,compare_config
    bert=root/'MSA Datasets/pretrained_berts/bert_en'
    ref=next(x for x in pc.read_json(PARENT/'provenance/bert_reference.json') if x['kind']=='bert_config')
    if compare_config(ref['config_reference'],read_config(bert/'config.json'))[0]:
        raise RuntimeError('BERT architecture/configuration changed')
    model=make_model(bert,a_dim,v_dim,pretrained=False)
    checkpoint=run/group/'seed_42/best.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False).to(device)

def fusion_predict(model,z,device,a_ix=None,v_ix=None,mode='TAV',zero=False,batch=64):
    n=len(z['y']);probs=[];logits=[]
    with torch.inference_mode():
        for start in range(0,n,batch):
            ix=np.arange(start,min(n,start+batch))
            ht=torch.as_tensor(z['ht'][ix],device=device)
            ha=torch.as_tensor(z['zero_ha'][ix] if zero else z['ha'][ix if a_ix is None else a_ix[ix]],device=device)
            hv=torch.as_tensor(z['hv'][ix if v_ix is None else v_ix[ix]],device=device)
            mask=torch.tensor(BITS[mode],device=device).expand(len(ix),-1)
            out=fuse(model,ht,ha,hv,mask)
            logits.append(out.cpu().numpy());probs.append(out.softmax(-1)[:,1].cpu().numpy())
    return np.concatenate(logits),np.concatenate(probs)

def encode_cache(model,data,path,signature,device,batch=8):
    meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        old=pc.read_json(meta)
        if old['signature']!=signature or pc.sha_file(path)!=old['sha256']:raise RuntimeError(f'Cache mismatch: {path}')
        return pc.load_npz(path)
    result=dict(ids=data['ids'],y=data['y'])
    buffers={k:[] for k in ('ht','ha','hv','zero_ha')+tuple('logits_'+c for c in MODES)}
    with torch.inference_mode():
        for start in range(0,len(data['y']),batch):
            end=start+batch
            tb=torch.as_tensor(data['text'][start:end],device=device)
            ta=torch.as_tensor(data['audio'][start:end],device=device)
            tv=torch.as_tensor(data['vision'][start:end],device=device)
            h=encode(model,tb,ta,tv)
            for key,value in zip(('ht','ha','hv'),h):buffers[key].append(value.cpu().numpy())
            buffers['zero_ha'].append(model.proj_a(torch.zeros_like(ta)).cpu().numpy())
            for c in MODES:
                logits=fuse(model,*h,torch.tensor(BITS[c],device=device).expand(len(tb),-1))
                buffers['logits_'+c].append(logits.cpu().numpy())
    result.update({k:np.concatenate(v) for k,v in buffers.items()})
    pc.atomic_npz(path,**result)
    pc.dump_json(meta,dict(signature=signature,sha256=pc.sha_file(path)))
    return result

def probabilities(logits):
    v=logits-np.max(logits,axis=1,keepdims=True);exp=np.exp(v);return exp[:,1]/exp.sum(1)

class EventResidual(torch.nn.Module):
    """The complete archived TRIM model stays frozen; only this small head is trained."""
    def __init__(self,event_dim,text_dim=256):
        super().__init__()
        self.delta=torch.nn.Sequential(torch.nn.Linear(event_dim,32),torch.nn.GELU(),torch.nn.Linear(32,1))
        self.gate=torch.nn.Sequential(torch.nn.Linear(text_dim,32),torch.nn.GELU(),torch.nn.Linear(32,1))
        torch.nn.init.zeros_(self.delta[-1].weight);torch.nn.init.zeros_(self.delta[-1].bias)
        torch.nn.init.constant_(self.gate[-1].bias,-2.)
    def forward(self,base_logits,ht,events,audio_present):
        d=2*torch.tanh(self.delta(events).squeeze(-1))*torch.sigmoid(self.gate(ht).squeeze(-1))
        d=d*audio_present.to(d.dtype)
        return base_logits+torch.stack((-d,d),dim=-1),d
