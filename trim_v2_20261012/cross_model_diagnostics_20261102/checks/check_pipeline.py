"""Engineering checks with synthetic inputs, never scientific results."""
from pathlib import Path
import argparse,hashlib,json,pickle,sys,tempfile
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from diag_common import cross_video,sequence_order,cases,compare,metrics,dump

def core():
 ids=np.asarray(['a$_$0','a$_$1','b$_$0','c$_$0','d$_$0','d$_$1'])
 for seed in range(50):
  d=cross_video(ids,seed);assert len(set(d))==len(ids);assert np.all(np.asarray([i.split('$_$')[0] for i in ids])[d]!=np.asarray([i.split('$_$')[0] for i in ids]))
 try:cross_video(np.asarray(['a$_$0','a$_$1','a$_$2','b$_$0']),0);raise AssertionError('Impossible bijection accepted')
 except ValueError:pass
 rng=np.random.default_rng(1);x=rng.normal(size=(len(ids),8,3)).astype('float32');x[:,[0,6,7]]=0
 y,n=sequence_order(x,ids,23);assert n==len(ids);assert np.array_equal(y,sequence_order(x,ids,23)[0]);assert np.allclose(x.sum(1),y.sum(1),atol=1e-6)
 defs,ds=cases(ids);assert len(defs)==82 and len(ds)==60
 for rep in range(20):
  da=ds[f'audio_cross_video_{rep:02}'][0];dv=ds[f'vision_cross_video_{rep:02}'][1];joint=ds[f'paired_av_cross_video_{rep:02}'];assert np.array_equal(joint[0],joint[1]);assert np.array_equal(da,joint[0])
  assert np.array_equal(ds[f'audio_cross_video_{rep:02}'][1],np.arange(len(ids)))
  assert np.array_equal(ds[f'vision_cross_video_{rep:02}'][0],np.arange(len(ids)))
 truth=np.array([0,0,1,1],np.int8);base=np.array([0,1,1,0],np.int8);new=np.array([1,0,1,0],np.int8)
 m=compare(truth,base,new,np.zeros(4),np.ones(4));assert m['changed']==2 and m['corrected']==1 and m['broken']==1 and m['f1_delta_pp']==0
 from summarize import bootstrap_effect
 b=np.stack([base]*5);z=np.stack([np.stack([new,new])]*5);lo,hi,nv=bootstrap_effect(truth,b,z,['a$_$0','b$_$0','c$_$0','d$_$0'],resamples=200)
 assert lo<=0<=hi and nv==4
 for fn in HERE.rglob('*.py'):compile(fn.read_text('utf8'),str(fn),'exec')
 result={'status':'passed','synthetic_only':True,'checks':['cross-video bijection across 50 draws','impossible donors rejected','deterministic temporal permutation','zero-row positions preserved','multiset sum preserved','82 declared conditions','paired donors preserve mutual A/V relation','corrected/broken identities','paired source-video bootstrap','Python syntax']}
 dump(HERE/'checks/core_check.json',result);print(json.dumps(result))

def model(method):
 import torch,yaml
 from transformers import BertConfig,BertModel
 torch.set_num_threads(2);torch.manual_seed(23);torch.backends.cudnn.enabled=False
 sys.path.insert(0,str(HERE/'_trim'));import model_tools as tm
 from adapters import Adapter
 check_folder=HERE/'checks';check_folder.mkdir(exist_ok=True)
 # TemporaryDirectory is verified inside this new bundle before its context cleanup.
 with tempfile.TemporaryDirectory(prefix='synthetic_',dir=str(check_folder)) as tmp:
  root=Path(tmp).resolve();assert root.is_relative_to(check_folder.resolve())
  bert=root/'MSA Datasets/pretrained_berts/bert_en';bert.mkdir(parents=True)
  cfg=BertConfig(vocab_size=128,hidden_size=768,num_hidden_layers=1,num_attention_heads=12,intermediate_size=128,max_position_embeddings=64,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1)
  cfg.save_pretrained(bert)
  vocab=[f'token{i}'for i in range(128)]
  for i,v in [(0,'[PAD]'),(100,'[UNK]'),(101,'[CLS]'),(102,'[SEP]'),(103,'[MASK]')]:vocab[i]=v
  (bert/'vocab.txt').write_text('\n'.join(vocab),encoding='utf8')
  rng=np.random.default_rng(17);n=6;ids=np.asarray([f'v{i//2}$_${i%2}' for i in range(n)]);labels=np.array([0,2,0,2,0,2]);reg=np.where(labels==2,.7,-.7).astype(np.float32)
  text=np.zeros((n,3,10),np.float32);text[:,0,:8]=rng.integers(3,80,(n,8));text[:,0,0]=101;text[:,0,7]=102;text[:,1,:8]=1
  split=dict(id=ids,classification_labels=labels,regression_labels=reg,text_bert=text,text=rng.normal(size=(n,10,768)).astype(np.float32),raw_text=['synthetic']*n)
  configs=None
  if method=='LNLN':
   configs=yaml.safe_load((HERE/'_lnln/configs/train_mosi.yaml').read_text('utf8'));configs['model']['feature_extractor']['input_length']=[10,16,12]
   split.update(audio=rng.normal(size=(n,12,5)).astype(np.float32),vision=rng.normal(size=(n,16,20)).astype(np.float32),audio_lengths=[10]*n,vision_lengths=[13]*n)
   split['audio'][:,10:]=0;split['vision'][:,13:]=0;name='unaligned_50.pkl'
  else:
   split.update(audio=rng.normal(size=(n,10,5)).astype(np.float32),vision=rng.normal(size=(n,10,20)).astype(np.float32));name='aligned_50.pkl'
   split['audio'][:,8:]=0;split['vision'][:,8:]=0
  data=root/'MSA Datasets/MOSI/Processed'/name;data.parent.mkdir(parents=True)
  with data.open('wb') as f:pickle.dump({'test':split},f,protocol=4)
  if method=='TRIM':native=tm.make_model(bert,5,20)
  elif method=='MulT':
   sys.path.insert(0,str(HERE/'_mult'));from run_cormult_repro import CorMulTReproModel
   native=CorMulTReproModel(768,5,20,d=30,corr_d_model=128,corr_out_dim=64,num_heads=2,layers=2,dropout=.1,out_dropout=0.,use_correlation=False,corr_model=None,mlp_head=False)
  else:
   sys.path.insert(0,str(HERE/'_lnln'));from models.lnln import build_model
   configs['model']['feature_extractor']['bert_pretrained']=str(bert)
   with patch.object(BertModel,'from_pretrained',side_effect=lambda *a,**kw:BertModel(cfg)):native=build_model(configs)
  ckpt=root/'synthetic.pt';torch.save(native.state_dict(),ckpt);del native
  if configs is None:adapter=Adapter(root,'MOSI',method,ckpt,torch.device('cpu'))
  else:
   with patch('yaml.safe_load',return_value=configs):adapter=Adapter(root,'MOSI',method,ckpt,torch.device('cpu'))
  definitions,donors=cases(adapter.ids,repetitions=2,temporal=2)
  def fingerprint():
   h=hashlib.sha256()
   for name,param in adapter.model.named_parameters():h.update(name.encode());h.update(param.detach().numpy().tobytes())
   return h.hexdigest()
  before=fingerprint();errors={};max_error=0.
  with torch.inference_mode():
   adapter.start(np.arange(4));errors=adapter.check_native(definitions,donors)
   for case in definitions:
    pred,score,got=adapter.predict(case,donors);aa,vv=adapter.inputs(case,donors)
    if method=='TRIM':
     ia=iv=adapter.ix;kind=case['kind']
     if kind in ['audio_cross_video','paired_av_cross_video']:ia=donors[case['name']][0][ia]
     if kind=='vision_cross_video':iv=donors[case['name']][1][iv]
     if kind=='paired_av_cross_video':iv=ia
     a=adapter.tensor(adapter.a[ia]);v=adapter.tensor(adapter.v[iv])
     if kind in ['zero_audio_present','zero_av_present']:a=torch.zeros_like(a)
     if kind in ['zero_vision_present','zero_av_present']:v=torch.zeros_like(v)
     adapter.model.use_audio=bool(case['mask'][1]);adapter.model.use_visual=bool(case['mask'][2])
     expected,_=adapter.model(adapter.batch_text[:,0],adapter.batch_text[:,1],adapter.batch_text[:,2],a,v,training=False)
     adapter.model.use_audio=True;adapter.model.use_visual=True
    elif method=='MulT':
     if not case['mask'][1]:aa=torch.zeros_like(aa)
     if not case['mask'][2]:vv=torch.zeros_like(vv)
     expected=adapter.model(adapter.batch_text,aa,vv)
    else:
     if not case['mask'][1]:aa=torch.zeros_like(aa)
     if not case['mask'][2]:vv=torch.zeros_like(vv)
     adapter.memo_enabled=False;expected=adapter.model((None,None,None),(vv,aa,adapter.batch_text))['sentiment_preds'];adapter.memo_enabled=True
    error=float((got-expected.cpu()).abs().max());max_error=max(max_error,error);assert error<1e-5,(case,error)
    assert not any(p.requires_grad for p in adapter.model.parameters())
  assert fingerprint()==before
  report=dict(status='passed',method=method,synthetic_only=True,conditions_checked=len(definitions),native_max_output_error=max_error,wrapper_checks=errors,
   memo_hits=adapter.memo_hits,memo_calls=adapter.memo_calls,parameters_unchanged=True,
   note='Reduced synthetic BERT depth and LNLN positional lengths; no trained model accuracy is measured. Real remote jobs separately require historical decision replay.')
  dump(HERE/'checks'/f'{method}_adapter_check.json',report);print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['TRIM','MulT','LNLN']);a=p.parse_args()
 if a.method:model(a.method)
 else:core()