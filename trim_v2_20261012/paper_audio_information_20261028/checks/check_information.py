"""Engineering checks on synthetic data only; remote training remains the user's workflow."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import information_core as core
import run_information as runner
from information_core import pc,np,torch
def synthetic():
 rng=np.random.default_rng(8);parts={};audios={};embeds={}
 for split,n in (('train',32),('valid',16)):
  t=12;d=4;y=np.arange(n,dtype=np.int8)%2
  a=np.zeros((n,t,d),dtype=np.float64);text=np.zeros((n,3,t),dtype=np.int64)
  for i in range(n):
   length=5+i%4;v=rng.normal(scale=.1,size=(length,d));v[:,0]+=(2*y[i]-1)*.7
   v[:,1]=1;v[:,2]+=np.linspace(-.5,.5,length)
   a[i,1:1+length]=v
   text[i,0,0]=101;text[i,0,1:1+length]=5;text[i,0,1+length]=102;text[i,1,:2+length]=1
  if split=='train':a[0,2:]=0
  ids=np.array([split+str(i//2)+'$_$'+str(i) for i in range(n)])
  parts[split]=dict(ids=ids,y=y,audio=a.mean(1).astype(np.float32),text=text)
  audios[split]=a;embeds[split]=rng.normal(size=(n,16)).astype(np.float32)
 return parts,audios,embeds
def main(out):
 if out.exists():raise RuntimeError('Use a new test output directory')
 out.mkdir(parents=True);core.runtime('cpu',2);core.verify_release();checks=[]
 parts,audios,embeds=synthetic()
 data,donors=core.prepare_arrays(parts,audios,embeds,out/'prepared',dict(synthetic=True))
 assert len(np.unique(donors))==32 and np.all(donors!=np.arange(32))
 quality=pc.read_json(out/'prepared/audio_quality.json')
 assert quality['quality'][0]['audio_text_mask_disagreement_count']==1
 assert data['train']['seq'].shape==(32,12,4) and data['train']['mask'].sum(1)[0]==1
 checks.append('audio_only_nonzero_proxy_compaction_preserves_anomalous_sample')
 before=pc.load_npz(out/'prepared/normalization.npz')
 changed_parts={s:dict(p) for s,p in parts.items()};changed_parts['valid']['text']=np.zeros_like(parts['valid']['text'])
 changed_audio={s:a.copy() for s,a in audios.items()};changed_audio['valid']*=100;changed_parts['valid']['audio']=changed_audio['valid'].mean(1).astype(np.float32)
 changed_emb={s:a.copy() for s,a in embeds.items()};changed_emb['valid']*=10000
 other,_=core.prepare_arrays(changed_parts,changed_audio,changed_emb,out/'changed_valid',dict(synthetic=True))
 after=pc.load_npz(out/'changed_valid/normalization.npz')
 for k in before:assert np.array_equal(before[k],after[k]),k
 for k in data['train']:assert np.array_equal(data['train'][k],other['train'][k]),k
 checks.append('all_normalization_uses_train_only_no_text_mask_or_validation_feature_leakage')
 seq=data['valid']['seq'];mask=data['valid']['mask'];ids=data['valid']['ids']
 ix=core.frame_indices(ids,mask,20261028);shuffled=core.frame_permute(seq,mask,ids,20261028)
 for i in range(len(ids)):
  valid=np.flatnonzero(mask[i]);invalid=np.flatnonzero(~mask[i])
  assert np.array_equal(ix[i,invalid],invalid)
  if len(valid)>1:assert np.all(ix[i,valid]!=valid)
  assert np.allclose(np.mean(seq[i,valid],0),np.mean(shuffled[i,valid],0),atol=1e-6)
  assert np.allclose(np.std(seq[i,valid],0),np.std(shuffled[i,valid],0),atol=1e-6)
 assert np.array_equal(shuffled,core.frame_permute(seq,mask,ids,20261028))
 cross=core.cross_video(ids,42);videos=np.array([s.split('$_$')[0] for s in ids])
 assert np.all(videos[cross]!=videos)
 checks.append('frame_order_interventions_preserve_masks_and_feature_multisets_cross_video_donors_valid')
 torch.manual_seed(42);cnn=core.Probe('temporal_cnn',4).eval()
 x=torch.as_tensor(seq);m=torch.as_tensor(mask)
 with torch.inference_mode():
  original=cnn(x,m)
  altered=x.clone();altered[~m]=1e5
  assert torch.allclose(original,cnn(altered,m),atol=1e-6)
  longer=torch.cat((x,torch.zeros(len(x),5,4)),1);long_mask=torch.cat((m,torch.zeros(len(x),5,dtype=torch.bool)),1)
  assert torch.allclose(original,cnn(longer,long_mask),atol=1e-6)
  assert torch.isfinite(cnn(torch.zeros_like(x),torch.zeros_like(m))).all()
  assert float((original-cnn(torch.as_tensor(shuffled),m)).abs().max())>1e-6
 checks.append('cnn_ignores_invalid_padding_values_and_has_nontrivial_order_sensitivity')
 torch.manual_seed(7);probe=core.Probe('observed_mean',4)
 toy=torch.zeros(64,4);labels=torch.arange(64)%2;toy[:,0]=labels.float()*4-2
 opt=torch.optim.AdamW(probe.parameters(),lr=.01)
 for _ in range(60):
  probe.train();opt.zero_grad();loss=torch.nn.functional.cross_entropy(probe(toy),labels)
  loss.backward();opt.step()
 probe.eval()
 with torch.inference_mode():assert float((probe(toy).argmax(1)==labels).float().mean())>.95
 checks.append('probe_learns_informative_synthetic_audio_feature')
 pipeline=out/'pipeline';pipeline.mkdir();pc.dump_json(pipeline/'pipeline_signature.json',dict(synthetic=True))
 pc.atomic_npz(pipeline/'normalization.npz',**before)
 pc.atomic_npz(pipeline/'data_index.npz',train_ids=data['train']['ids'],valid_ids=data['valid']['ids'],label_shuffle_train_donors=donors)
 args=SimpleNamespace(out=pipeline,device=torch.device('cpu'),epochs=2,batch=8,permutations=2,seeds=[42])
 train_labels=data['train']['y'].copy();valid_labels=data['valid']['y'].copy();jobs=[]
 for recipe in core.RECIPES:
  jobs.append(runner.fit(args,data,recipe,42,donors))
 assert np.array_equal(data['train']['y'],train_labels) and np.array_equal(data['valid']['y'],valid_labels)
 assert runner.fit(args,data,core.RECIPES[0],42,donors)==jobs[0]
 runner.summarize(args,data,jobs)
 checks.append('all_eight_probe_jobs_checkpoint_replay_controls_train_label_shuffle_only_and_resume_cache')
 with zipfile.ZipFile(pipeline/'audio_information_review_20261028.zip') as z:
  assert z.testzip() is None
  manifest=json.loads(z.read('audio_information_review/review_manifest.json'))
  for rel,digest in manifest['files'].items():
   assert hashlib.sha256(z.read('audio_information_review/'+rel)).hexdigest()==digest,rel
  assert not any(p.endswith(('dataset_train.npz','dataset_valid.npz')) for p in z.namelist())
 checks.append('review_metrics_predictions_small_weights_hash_verified_and_large_data_excluded')
 report=dict(status='passed',synthetic=True,test_evaluated=False,checks=checks,torch=torch.__version__,
  tested_sources={name:pc.sha_file(core.HERE/name) for name in ('information_core.py','run_information.py','remote_audio_information.sh')},
  note='Synthetic engineering tests only. Real model training is scheduled for the remote server, not performed locally.')
 pc.dump_json(out/'engineering_check.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.out)