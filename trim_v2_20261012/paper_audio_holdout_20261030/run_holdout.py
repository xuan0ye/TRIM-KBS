"""Remote fresh-expert training with video-disjoint routing holdouts."""
import argparse,zipfile
from holdout_router import *

def package_review(args,summary):
 pc.dump_json(args.out/'summary.json',summary)
 # Only compact experts, readouts and selected routers; never the large TRIM weights or optimizer resumes.
 files=[]
 for path in args.out.rglob('*'):
  if not path.is_file() or path.suffix not in ('.json','.csv','.npz','.pt'):continue
  rel=path.relative_to(args.out)
  if path.name in ('review_manifest.json','resume.pt') or (path.suffix=='.pt' and 'trim_expert' in rel.parts):continue
  files.append(path)
 sources={'PROTOCOL.md':HERE/'PROTOCOL.md','source_manifest.json':HERE/'bundle_manifest.json',
  'holdout_split.py':HERE/'holdout_split.py','holdout_core.py':HERE/'holdout_core.py','holdout_router.py':HERE/'holdout_router.py','run_holdout.py':HERE/'run_holdout.py'}
 manifest={p.relative_to(args.out).as_posix():pc.sha_file(p) for p in files};manifest.update({k:pc.sha_file(v) for k,v in sources.items()})
 pc.dump_json(args.out/'review_manifest.json',dict(files=manifest,test_evaluated=False))
 archive=args.out/'audio_holdout_review_20261030.zip';temp=archive.with_suffix('.tmp.zip')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[args.out/'review_manifest.json']:z.write(p,'audio_holdout_review/'+p.relative_to(args.out).as_posix())
  for name,path in sources.items():z.write(path,'audio_holdout_review/'+name)
 os.replace(temp,archive);print('AUDIO_HOLDOUT_REVIEW_COMPLETE',archive,flush=True)

def run_pipeline(args):
 release=verify_release();args.out.mkdir(parents=True,exist_ok=True);args.device=pt.runtime(args.device,4)
 data,input_signature=prepare_training(args)
 signature=dict(release=release,inputs=input_signature,seed=args.seed,paper_phases=[5,8,7],paper_batch=args.batch,paper_acc_steps=args.acc_steps,
  audio_epochs=15,router_epochs=args.router_epochs,permutations=args.permutations,torch=torch.__version__,numpy=np.__version__,
  fresh_supervised_experts=True,external_used_for_selection=False,test_evaluated=False)
 pc.record_json(args.out/'pipeline_signature.json',signature)
 print('HOLDOUT_STAGE original_pretrained_bert_verification',flush=True)
 bert_folder=args.out/'bert_preflight';bert_folder.mkdir(exist_ok=True);bert=preflight_bert(args.root,bert_folder)
 paper_args=SimpleNamespace(out=args.out/'trim_expert',smoke=False,batch=args.batch,acc_steps=args.acc_steps,device=args.device)
 inputs=dict(pipeline_sha256=pc.sha_file(args.out/'pipeline_signature.json'),split_sha256=pc.sha_file(HERE/'split_plan.json'),
  original_bert_sha256=pc.sha_file(bert_folder/'bert_verified.json'))
 print('HOLDOUT_STAGE fresh_paper_TRIM_train_select_expert_dev_only',flush=True)
 pt.run_job(paper_args,'paper_original',args.seed,expert_parts(data),bert,inputs)
 print('HOLDOUT_STAGE fresh_audio_classifier_train_select_expert_dev_only',flush=True)
 audio=train_audio(args,data['expert_train'],data['expert_dev'],signature)
 trim=load_trim(args,bert,data)
 read={r:readouts(args,r,data[r],trim,audio,signature) for r in ROLES}
 # Compact audio neural replay samples; no full training embedding table is exported.
 feature_samples={}
 for r in ROLES:
  n=min(64,len(data[r]['x']));feature_samples[r+'_x']=data[r]['x'][:n];feature_samples[r+'_ids']=data[r]['paper']['ids'][:n]
 jobs={}
 for kind in CONDITIONS:jobs[kind]=train_router(args,kind,read['router_train'],read['router_dev'],signature)
 trim_path=args.out/'trim_expert/paper_original'/f'seed_{args.seed}/best.pt';audio_path=args.out/'audio_expert/best.pt'
 seal=dict(pipeline_sha256=pc.sha_file(args.out/'pipeline_signature.json'),trim_sha256=pc.sha_file(trim_path),audio_sha256=pc.sha_file(audio_path),
  expert_selection_role='expert_dev',router_selection_role='router_dev',routers={k:pc.sha_file(args.out/'routers'/k/'best.pt') for k in CONDITIONS},
  external_labels_used_for_selection=False,test_evaluated=False)
 pc.record_json(args.out/'selection_seal.json',seal)
 print('HOLDOUT_SELECTIONS_SEALED external_validation_now_allowed',flush=True)
 external_data=prepare_external(args)
 external=readouts(args,'external_valid',external_data,trim,audio,signature)
 feature_samples['external_valid_x']=external_data['x'];feature_samples['external_valid_ids']=external_data['paper']['ids']
 pc.atomic_npz(args.out/'audio_replay_features.npz',**feature_samples)
 evaluation=external_evaluation(args,external,jobs,seal)
 if pc.sha_file(trim_path)!=seal['trim_sha256'] or pc.sha_file(audio_path)!=seal['audio_sha256']:raise RuntimeError('Expert weights changed during routing/evaluation')
 old=pc.load_npz(args.parent_run/'paper_original/seed_42/validation_predictions.npz')
 if not np.array_equal(old['ids'],external['ids']):raise RuntimeError('Original baseline external IDs differ')
 reference=pc.scores(external['y'],old['pred_TAV'],old['prob_TAV'])
 internal={r:dict(n=len(z['y']),trim={m:scores(z['y'],z['logits_'+m]) for m in MODES},audio=scores(z['y'],z['audio_logits'])) for r,z in read.items()}
 summary=dict(status='complete',scope='single_split_fresh_experts_video_holdout_router_pilot',pipeline=signature,roles=pc.read_json(HERE/'split_plan.json'),
  internal=internal,external=evaluation,old_all_training_rows_parent_reference=reference,
  old_reference_caveat='Old parent fitted 6065 rows and selected on this validation set; new expert fitted only 4208 rows with independent expert_dev selection, so this reference is not a paired routing control',
  selections=seal,router_jobs=jobs,test_evaluated=False,
  caveats=['Original external validation has been reused across earlier studies; this remains exploratory, not an independent test',
   'This is one video-disjoint holdout split, not OOF and not three independent backbone runs',
   'Compare gates first against the same newly trained TRIM expert; report the old full-training reference separately',
   'Media availability subset remains biased; PANNs combines speech/prosody/environment, not isolated environmental sound'])
 pc.dump_json(args.out/'suite_progress.json',dict(status='complete',external_evaluated=True,test_evaluated=False))
 package_review(args,summary);print('AUDIO_HOLDOUT_SUITE_COMPLETE test_evaluated=False',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE.parent.parent)
 ap.add_argument('--parent-run',type=Path,default=PARENT/'runs/pilot');ap.add_argument('--out',type=Path,default=HERE/'runs/holdout')
 ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--acc-steps',type=int,default=4)
 args=ap.parse_args();args.seed=42;args.router_epochs=50;args.permutations=20
 if args.batch<2 or args.acc_steps<1:raise ValueError('batch>=2, acc_steps>=1 required')
 run_pipeline(args)
if __name__=='__main__':main()