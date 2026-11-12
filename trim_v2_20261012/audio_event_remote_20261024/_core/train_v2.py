"""Train TRIM-v2 candidates. Pilot is validation-only; testing requires an explicit flag."""
from pathlib import Path
import argparse,gc,json,math,os,time
import numpy as np
import torch,transformers
from transformers import get_cosine_schedule_with_warmup
from common import sha_file,dump_json,write_csv,SEEDS,DATASETS
from model_v2 import TRIMV2,VARIANTS,objective
from data_v2 import *
from runtime_v2 import *
from evaluate_v2 import fixed_predictions,interventions,seed_metrics,validation,scores
HERE=Path(__file__).resolve().parent
def log(*x):print(time.strftime("%Y-%m-%dT%H:%M:%S"),*x,flush=True)
def job_signature(args,d,v,seed,phases,data_sha,bert_sha,stats):
 files=("train_v2.py","model_v2.py","data_v2.py","runtime_v2.py","evaluate_v2.py","common.py","aggregate_core.py","mmap_dataset_pickle.py")
 return json.loads(json.dumps(dict(dataset=d,variant=v,seed=seed,view=args.view,phases=phases,batch=args.batch,acc_steps=args.acc_steps,d_model=128,latent_tokens=8,heads=4,uni_weight=.2,kd_weight=.1,kd_margin=.02,kd_temperature=2.,validation_criterion="mean_macro_f1_T_TA_TV_TAV",data_sha256=data_sha,bert_weights_sha256=bert_sha,bert_config_sha256=sha_file(args.root/"MSA Datasets/pretrained_berts"/("bert_cn" if d=="SIMS" else "bert_en")/"config.json"),train_statistics=stats,torch=torch.__version__,transformers=transformers.__version__,smoke=args.smoke,code_sha256={f:sha_file(HERE/f) for f in files})))
def selected_test(model,args,d,v,seed,parts,folder,signature,best):
 path=folder/"test_complete.json"
 if path.exists():
  saved=json.loads(path.read_text(encoding="utf8"))
  assert saved["signature"]==signature and saved["prediction_sha256"]==sha_file(folder/"test_predictions.npz")
  log("VERIFIED_TEST_CACHE",d,v,seed);return
 pred,prob,reps=fixed_predictions(model,parts["test"],args.batch,args.device,cache_representations=True)
 pert=interventions(model,reps,max(args.batch,32),args.device)
 arrays=dict(ids=parts["test"].ids,labels=parts["test"].y,**pred,**{"prob_"+k:z for k,z in prob.items()},**pert)
 temp=folder/"test_predictions.tmp.npz";np.savez_compressed(temp,**arrays);os.replace(temp,folder/"test_predictions.npz")
 rows,masks=seed_metrics(parts["test"],pred,prob,dataset=d,variant=v,seed=seed)
 write_csv(folder/"test_seed_metrics.csv",rows);write_csv(folder/"test_mask_metrics.csv",masks)
 dump_json(path,dict(signature=signature,best_validation=best,prediction_sha256=sha_file(folder/"test_predictions.npz"),checkpoint_sha256=sha_file(folder/"best.pt"),status="complete",test_used_for_selection=False))
 log("TEST_EVALUATION_DONE",d,v,seed,"complete_f1",round(scores(parts["test"].y,pred["full"],prob["full"])["macro_f1"],6))
def run_job(args,d,v,seed,parts,bert,data_sha,bert_sha,stats):
 folder=args.out/d/v/f"seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
 phases=(1,1,0) if args.smoke else ((5,8,0) if d=="SIMS" else (5,8,7))
 signature=job_signature(args,d,v,seed,phases,data_sha,bert_sha,stats)
 complete=folder/"training_complete.json";resume_file=folder/"resume.pt";resume=None
 cached=None
 if complete.exists():
  cached=json.loads(complete.read_text(encoding="utf8"))
  if cached["signature"]!=signature:raise RuntimeError(f"Configuration changed: {folder}; use a new output directory.")
  assert cached["checkpoint_sha256"]==sha_file(folder/"best.pt")
  if not args.evaluate_test:log("VERIFIED_TRAIN_CACHE",d,v,seed);return "complete"
  testfile=folder/"test_complete.json"
  if testfile.exists():
   rec=json.loads(testfile.read_text(encoding="utf8"));assert rec["signature"]==signature and rec["prediction_sha256"]==sha_file(folder/"test_predictions.npz")
   log("VERIFIED_TRAIN_AND_TEST_CACHE",d,v,seed);return "complete"
 seed_everything(seed)
 if cached is None and resume_file.exists():
  resume=torch.load(resume_file,map_location="cpu",weights_only=False)
  if resume["signature"]!=signature:raise RuntimeError(f"Resume configuration changed: {folder}; use a new output directory.")
 model=TRIMV2(bert,parts["train"].a_dim,parts["train"].v_dim,variant=v,pretrained=resume is None and cached is None).to(args.device);model.set_feature_stats(stats)
 if cached is not None:
  model.load_state_dict(torch.load(folder/"best.pt",map_location="cpu",weights_only=True,mmap=True),strict=True);best=cached["best_validation"]
  selected_test(model,args,d,v,seed,parts,folder,signature,best);del model;gc.collect()
  if args.device.type=="cuda":torch.cuda.empty_cache()
  return "complete"
 if resume is not None:model.load_state_dict(resume.pop("model"),strict=True)
 generator=torch.Generator().manual_seed(seed);tr_loader=make_loader(parts["train"],args.batch,shuffle=True,generator=generator)
 counts=np.bincount(parts["train"].y,minlength=2);weights=np.sqrt(counts.sum()/(2*counts));weights/=weights.mean()
 weights=torch.tensor(weights,dtype=torch.float32,device=args.device)
 best=resume["best"] if resume is not None else dict(selection_score=-1.)
 history=resume["history"] if resume is not None else [];began=time.monotonic()
 log("V2_TRAIN_START",d,v,seed,"view",args.view,"phases",phases,"n_train",len(parts["train"]),"cudnn",torch.backends.cudnn.enabled)
 for phase,epochs in enumerate(phases):
  if epochs==0 or (resume is not None and phase<resume["phase"]):continue
  configure_phase(model,phase);optimizer=optimizer_for(model,phase);steps=math.ceil(len(tr_loader)/args.acc_steps)
  scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(steps*epochs*.1)),steps*epochs)
  first=1;phase_best=-1.;phase_file=folder/"phase_best.pt"
  if resume is not None and phase==resume["phase"]:
   optimizer.load_state_dict(resume["optimizer"]);scheduler.load_state_dict(resume["scheduler"]);restore_rng(resume["rng"],generator)
   first=resume["epoch"]+1;phase_best=resume["phase_best"]
  for epoch in range(first,epochs+1):
   tick=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);accum={k:0. for k in ("loss","subset_ce","unimodal_ce","consistency","teacher_selected")}
   for i,data in enumerate(tr_loader):
    data=to_device(data,args.device);components=model.components(data)
    kd_weight=.1*min(1.,(len(history)+1)/3.)
    loss,detail=objective(model,components,data["y"],weights,kd_weight=kd_weight)
    if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite loss: {d}/{v}/{seed}/P{phase+1}/E{epoch}")
    group_start=(i//args.acc_steps)*args.acc_steps;group_size=min(args.acc_steps,len(tr_loader)-group_start)
    (loss/group_size).backward()
    if (i+1)%args.acc_steps==0 or i+1==len(tr_loader):
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    accum["loss"]+=float(loss.detach().cpu())
    for k in ("subset_ce","unimodal_ce","consistency","teacher_selected"):accum[k]+=float(detail[k])
   val=validation(model,parts["valid"],args.batch,args.device);score=val["selection_score"]
   if score>phase_best:phase_best=score;save_torch(phase_file,cpu_state(model))
   if score>best["selection_score"]:
    best=dict(**val,phase=phase+1,epoch=epoch);save_torch(folder/"best.pt",cpu_state(model));dump_json(folder/"best_validation.json",best)
   seconds=time.monotonic()-tick
   row=dict(phase=phase+1,epoch=epoch,**{k:z/len(tr_loader) for k,z in accum.items()},val_score=score,val_complete_f1=val["complete_macro_f1"],val_text_f1=val["text_macro_f1"],seconds=round(seconds,2),optimizer_steps=steps)
   history.append(row);write_csv(folder/"validation_history.csv",history)
   save_torch(resume_file,dict(signature=signature,phase=phase,epoch=epoch,phase_best=phase_best,best=best,history=history,model=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator)))
   dump_json(args.out/"current_job.json",dict(dataset=d,variant=v,seed=seed,phase=phase+1,epoch=epoch,phase_epochs=epochs,epoch_seconds=round(seconds,1),phase_remaining_minutes=round(seconds*(epochs-epoch)/60,1),best_validation_score=best["selection_score"],status="training"))
   log("EPOCH",d,v,seed,"P",phase+1,f"{epoch}/{epochs}","val_subsets",round(score,6),"val_full",round(val["complete_macro_f1"],6),"seconds",round(seconds,1),"phase_eta_min",round(seconds*(epochs-epoch)/60,1))
   if args.smoke and args.smoke_stop_after_epoch and len(history)>=args.smoke_stop_after_epoch:
    log("SMOKE_PAUSED_AFTER_SAVED_EPOCH");return "paused"
  model.load_state_dict(torch.load(phase_file,map_location="cpu",weights_only=True,mmap=True),strict=True)
  resume=None;del optimizer,scheduler;gc.collect()
 model.load_state_dict(torch.load(folder/"best.pt",map_location="cpu",weights_only=True,mmap=True),strict=True)
 verified=validation(model,parts["valid"],args.batch,args.device);assert abs(verified["selection_score"]-best["selection_score"])<1e-12
 dump_json(complete,dict(signature=signature,best_validation=best,checkpoint_sha256=sha_file(folder/"best.pt"),elapsed_seconds=round(time.monotonic()-began,1),status="complete",test_evaluated=False))
 resume_file.unlink(missing_ok=True);phase_file.unlink(missing_ok=True)
 if args.evaluate_test:selected_test(model,args,d,v,seed,parts,folder,signature,best)
 log("V2_TRAIN_DONE",d,v,seed,"minutes",round((time.monotonic()-began)/60,1),"best_val_subsets",round(best["selection_score"],6))
 del model;gc.collect()
 if args.device.type=="cuda":torch.cuda.empty_cache()
 return "complete"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=HERE.parent);ap.add_argument("--out",type=Path,default=None)
 ap.add_argument("--datasets",nargs="+",choices=DATASETS,default=["MOSI","SIMS"]);ap.add_argument("--variants",nargs="+",choices=VARIANTS,default=["text","mean","v2","no_uni"]);ap.add_argument("--seeds",nargs="+",type=int,default=[42])
 ap.add_argument("--view",choices=["native","aligned"],default="native");ap.add_argument("--batch",type=int,default=8);ap.add_argument("--acc-steps",type=int,default=4);ap.add_argument("--device",default="auto");ap.add_argument("--threads",type=int,default=4)
 ap.add_argument("--evaluate-test",action="store_true");ap.add_argument("--smoke",action="store_true");ap.add_argument("--smoke-stop-after-epoch",type=int,default=0)
 args=ap.parse_args();args.device=runtime(args.device,args.threads);args.out=args.out or HERE/("smoke_results" if args.smoke else ("experiments" if args.view=="native" else "experiments_aligned"));args.out.mkdir(parents=True,exist_ok=True)
 if args.batch<2 or args.acc_steps<1:raise ValueError("batch>=2 and acc-steps>=1 required")
 os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
 plan=dict(datasets=args.datasets,variants=args.variants,seeds=args.seeds,view=args.view,smoke=args.smoke,evaluate_test=args.evaluate_test,selection="mean Macro-F1 of four validation subsets",torch=torch.__version__,transformers=transformers.__version__,cudnn_enabled=torch.backends.cudnn.enabled)
 dump_json(args.out/"suite_plan.json",plan);done=0;total=len(args.datasets)*len(args.variants)*len(args.seeds);dump_json(args.out/"progress.json",dict(done=0,total=total,status="loading"))
 for d in args.datasets:
  file=data_path(args.root,d,args.view);log("LOAD_SEQUENCE_DATA",d,file);data_sha=sha_file(file);raw=read_dataset(args.root,d,args.view)
  fulltrain=SequenceDataset(raw,d,"train",args.view);stats_file=args.out/f"train_stats_{d}_{args.view}.json"
  if stats_file.exists():
   stored=json.loads(stats_file.read_text(encoding="utf8"));assert stored["data_sha256"]==data_sha;stats=stored["statistics"]
  else:
   stats=train_statistics(fulltrain);dump_json(stats_file,dict(data_sha256=data_sha,statistics=stats))
  parts={s:SequenceDataset(raw,d,s,args.view,limit=16 if args.smoke else None) for s in ("train","valid","test")};del fulltrain
  for left,right in (("train","valid"),("train","test"),("valid","test")):assert not set(parts[left].ids)&set(parts[right].ids)
  reference=HERE/"reference_ids"/f"{d}.npz"
  if not args.smoke:
   if not reference.exists():raise FileNotFoundError(f"Required comparison IDs missing: {reference}")
   with np.load(reference,allow_pickle=False) as z:assert np.array_equal(z["ids"],parts["test"].ids) and np.array_equal(z["labels"],parts["test"].y)
  bert=args.root/"MSA Datasets/pretrained_berts"/("bert_cn" if d=="SIMS" else "bert_en");weight=bert/"model.safetensors"
  if not weight.exists():weight=bert/"pytorch_model.bin"
  bert_sha=sha_file(weight)
  for v in args.variants:
   for seed in args.seeds:
    status=run_job(args,d,v,seed,parts,bert,data_sha,bert_sha,stats)
    if status=="paused":dump_json(args.out/"progress.json",dict(done=done,total=total,status="paused_smoke"));return
    done+=1;dump_json(args.out/"progress.json",dict(done=done,total=total,last=f"{d}/{v}/seed_{seed}",status="running"))
  del raw,parts;gc.collect()
 dump_json(args.out/"progress.json",dict(done=done,total=total,status="complete"));log("TRIM_V2_SUITE_COMPLETE",done,"jobs","test_evaluated",args.evaluate_test)
if __name__=="__main__":main()
