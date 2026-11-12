"""Exercise the complete remote diagnosis path on synthetic inputs and real saved weights."""
from pathlib import Path
from types import SimpleNamespace
import io,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from expert_core import *

def main():
 info.runtime('cpu',2);rng=np.random.default_rng(29)
 scratch=HERE.parent.parent/'.tmp/expert_engineering_20261030';scratch.mkdir(parents=True,exist_ok=True)
 out=Path(tempfile.mkdtemp(prefix='diagnose_',dir=scratch))
 args=SimpleNamespace(out=out,device=torch.device('cpu'),fusion_run=FUSION/'remote_results/review_20261030')
 data={};contexts={}
 for split,n in (('train',32),('valid',16)):
  data[split]=dict(ids=np.array([split+str(i) for i in range(n)]),y=np.tile([0,1],n//2).astype(np.int64),x=rng.normal(0,.4,(n,2048)).astype(np.float32))
  contexts[split]=dict(ht=rng.normal(0,.1,(n,256)).astype(np.float32),**{'logits_'+m:rng.normal(0,.4,(n,2)).astype(np.float32) for m in MODES})
 with zipfile.ZipFile(HERE.parent/'audio_information_review_20261028.zip') as z:
  warm=torch.load(io.BytesIO(z.read('audio_information_review/panns_embedding/seed_13/best.pt')),map_location='cpu',weights_only=True)
 result=diagnose(args,data,contexts,{13:warm})
 assert result['status']=='complete' and len(result['audio'])==2 and result['gradient_summary']['rows']==2
 for f in ('gradient_diagnostics.csv','gradient_sample.npz','train_readouts.npz','mechanism_diagnosis.json'):assert (out/f).is_file()
 assert pc.load_npz(out/'gradient_sample.npz')['x'].shape==(32,2048)
 assert pc.load_npz(out/'train_readouts.npz')['warm_audio_logits_13'].shape==(32,2)
 pc.dump_json(HERE/'checks/diagnosis_smoke.json',dict(status='pass',scope='synthetic_features_real_preexisting_weights_no_optimizer',
  training_rows=32,validation_rows=16,gradient_snapshots=2,tested_core_sha256=pc.sha_file(HERE/'expert_core.py'),test_evaluated=False))
 print('EXPERT_DIAGNOSIS_SMOKE_PASS',flush=True)
if __name__=='__main__':main()