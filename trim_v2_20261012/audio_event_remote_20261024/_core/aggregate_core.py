"""Aggregate paired core controls; never pool mask replicates as independent training runs."""
from pathlib import Path
import argparse,json,math
import numpy as np
from common import *
HERE=Path(__file__).resolve().parent
METRICS=("accuracy","macro_f1","balanced_accuracy","mcc","weighted_f1","brier","nll","ece15")
CONDITIONS=("complete","T","TA","TV")+tuple(f"p={p:.1f}" for p in RATES)
KEYS={"complete":"full","T":"text_only","TA":"text_audio","TV":"text_visual"}
MASK_ORDER=("text_only","text_audio","text_visual","full")
PERTURB=("shuffle_audio","shuffle_visual","shuffle_both","zero_audio_present","zero_visual_present","zero_both_present")
def scores(y,p,pr):
 out=metric(y,p);pr=np.asarray(pr,dtype=float)
 assert np.isfinite(pr).all() and np.all((pr>=0)&(pr<=1))
 out["brier"]=float(np.mean((pr-y)**2))
 clipped=np.clip(pr,1e-12,1-1e-12)
 out["nll"]=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped)))
 confidence=np.maximum(pr,1-pr);correct=(p==y)
 bins=np.minimum((confidence*15).astype(int),14);ece=0.
 for b in range(15):
  ix=bins==b
  if ix.any():ece+=float(ix.mean()*abs(correct[ix].mean()-confidence[ix].mean()))
 out["ece15"]=ece
 return out
def stats(vals):
 a=np.asarray(vals,dtype=float)
 return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),n=int(len(a)),values=a.tolist())
def aggregate(args):
 expected=expected_index();seedrows=[];maskrows=[];responses=[];probes=[];sources=[]
 for d in DATASETS:
  reference=None
  for v in VARIANTS:
   for seed in SEEDS:
    file=args.out/"predictions"/d/v/f"seed_{seed}.npz"
    meta=json.loads(file.with_suffix(".json").read_text(encoding="utf8"))
    assert meta["signature"]["dataset"]==d and meta["signature"]["variant"]==v and meta["signature"]["seed"]==seed
    assert meta["prediction_sha256"]==sha_file(file),str(file)
    sources.append(dict(file=str(file),sha256=meta["prediction_sha256"],signature=meta["signature"]))
    with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    y=a["labels"]
    if reference is None:reference=(a["ids"],y)
    assert np.array_equal(a["ids"],reference[0]) and np.array_equal(y,reference[1])
    complete=scores(y,a["full"],a["prob_full"])
    for key in ("accuracy","macro_f1"):
     assert abs(complete[key]-expected[d,v,seed][key])<1e-10
    for condition,key in KEYS.items():
     seedrows.append(dict(dataset=d,variant=v,seed=seed,condition=condition,n_masks=0,**scores(y,a[key],a["prob_"+key])))
    pp=np.stack([a[k] for k in MASK_ORDER]);pr=np.stack([a["prob_"+k] for k in MASK_ORDER]);ii=np.arange(len(y))
    for p in RATES:
     per=[]
     for ms in MASK_SEEDS:
      ix=draw_masks(len(y),p,ms);sc=scores(y,pp[ix,ii],pr[ix,ii]);per.append(sc)
      maskrows.append(dict(dataset=d,variant=v,seed=seed,condition=f"p={p:.1f}",mask_seed=ms,**sc))
     seedrows.append(dict(dataset=d,variant=v,seed=seed,condition=f"p={p:.1f}",n_masks=len(MASK_SEEDS),n_test=len(y),**{k:float(np.mean([s[k] for s in per])) for k in METRICS}))
    if v=="text_only":
     for k in tuple(MASK_ORDER)+PERTURB:
      assert np.array_equal(a[k],a["full"]),("Legacy text model changed labels",d,seed,k)
      assert np.allclose(a["prob_"+k],a["prob_full"],rtol=0,atol=1e-6),("Legacy text model changed probabilities",d,seed,k)
    base=a["full"];bp=a["prob_full"];correct=base==y
    for case in ("text_only","text_audio","text_visual")+PERTURB:
     cp=a[case];delta=a["prob_"+case]-bp
     responses.append(dict(dataset=d,variant=v,seed=seed,intervention=case,n_test=len(y),changed_labels=int(np.count_nonzero(cp!=base)),wrong_to_right=int(np.count_nonzero(~correct&(cp==y))),right_to_wrong=int(np.count_nonzero(correct&(cp!=y))),mean_abs_positive_probability_delta=float(np.mean(abs(delta))),max_abs_positive_probability_delta=float(np.max(abs(delta))),macro_f1_delta_pp=100*(metric(y,cp)["macro_f1"]-complete["macro_f1"])))
    if meta["diagnostic_probe"] is not None:probes.append(dict(dataset=d,variant=v,seed=seed,**meta["diagnostic_probe"]))
 summaries=[]
 for d in DATASETS:
  for v in VARIANTS:
   for c in CONDITIONS:
    rows=[r for r in seedrows if (r["dataset"],r["variant"],r["condition"])==(d,v,c)]
    assert len(rows)==5
    row=dict(dataset=d,variant=v,condition=c,n_training_seeds=5,n_masks_per_seed=10 if c.startswith("p=") else 0)
    for k in METRICS:
     st=stats([r[k] for r in rows]);row[k+"_mean"]=st["mean"];row[k+"_sd"]=st["sd"]
    summaries.append(row)
 index={(r["dataset"],r["variant"],r["seed"],r["condition"]):r for r in seedrows};paired=[]
 for d in DATASETS:
  for c in CONDITIONS:
   for baseline in ("no_modrop","text_only"):
    row=dict(dataset=d,contrast="full_minus_"+baseline,condition=c)
    for k in METRICS:
     vals=[index[d,"full",s,c][k]-index[d,baseline,s,c][k] for s in SEEDS]
     st=stats(vals);row[k+"_mean_delta"]=st["mean"];row[k+"_sd_delta"]=st["sd"]
     if k=="macro_f1":row.update(macro_f1_delta_pp=100*st["mean"],seed_deltas_pp=[100*x for x in vals],positive_seeds=sum(x>1e-12 for x in vals),negative_seeds=sum(x< -1e-12 for x in vals))
    paired.append(row)
 interactions=[]
 for d in DATASETS:
  for p in RATES:
   c=f"p={p:.1f}";vals=[]
   for s in SEEDS:
    dropfull=index[d,"full",s,c]["macro_f1"]-index[d,"full",s,"complete"]["macro_f1"]
    dropnone=index[d,"no_modrop",s,c]["macro_f1"]-index[d,"no_modrop",s,"complete"]["macro_f1"]
    vals.append(100*(dropfull-dropnone))
   interactions.append(dict(dataset=d,condition=c,contrast="(full_missing-full_complete)-(no_modrop_missing-no_modrop_complete)",unit="percentage points",**stats(vals)))
 intervention_summary=[]
 for d in DATASETS:
  for v in VARIANTS:
   for case in ("text_only","text_audio","text_visual")+PERTURB:
    rr=[r for r in responses if (r["dataset"],r["variant"],r["intervention"])==(d,v,case)]
    row=dict(dataset=d,variant=v,intervention=case,total_sample_seed_pairs=sum(r["n_test"] for r in rr),total_changed_labels=sum(r["changed_labels"] for r in rr),total_wrong_to_right=sum(r["wrong_to_right"] for r in rr),total_right_to_wrong=sum(r["right_to_wrong"] for r in rr))
    for key in ("mean_abs_positive_probability_delta","macro_f1_delta_pp"):
     st=stats([r[key] for r in rr]);row[key+"_mean"]=st["mean"];row[key+"_sd"]=st["sd"]
    row["max_abs_positive_probability_delta"]=max(r["max_abs_positive_probability_delta"] for r in rr)
    intervention_summary.append(row)
 write_csv(args.out/"core_seed_metrics.csv",seedrows);write_csv(args.out/"core_mask_metrics.csv",maskrows)
 write_csv(args.out/"core_summary.csv",summaries);write_csv(args.out/"paired_core_comparisons.csv",paired)
 write_csv(args.out/"robustness_interaction.csv",interactions);write_csv(args.out/"intervention_seed_metrics.csv",responses)
 write_csv(args.out/"intervention_summary.csv",intervention_summary)
 report=dict(metadata=dict(n_checkpoints=len(sources),n_seed_conditions=len(seedrows),n_mask_records=len(maskrows),n_probe_records=len(probes),training_seeds=SEEDS,mask_seeds=MASK_SEEDS,rates=RATES,metric_units="Fractions; MCC in [-1,1]. Explicit *_pp fields are percentage points.",aggregation="Confusion/metric per mask; mean 10 masks within each training seed; mean and sample SD over five seeds.",text_only_note="Independently trained historical text-only variant retains the fusion stack and auxiliary losses on missing placeholders. This is not a plain BERT classifier.",probe_note="Train-split gradient probes in eval mode, BERT frozen, no weight updates. They cannot establish historical training dynamics.",calibration_note="Uncalibrated softmax outputs: positive-class Brier and binary NLL; confidence ECE with 15 fixed equal-width bins on [0,1]. No temperature fitted.",test_selection="None: all prespecified five seeds and existing checkpoints retained; new diagnostics do not select a best test run."),summary=summaries,paired=paired,robustness_interaction=interactions,intervention_summary=intervention_summary,diagnostic_probes=probes,sources=sources)
 dump_json(args.out/"core_results.json",report)
 make_table(args.out,summaries)
 dump_json(args.out/"aggregate_checks.json",dict(status="complete",checkpoint_count=len(sources),complete_input_matches_all_original_csvs=True,legacy_text_label_invariance=True,legacy_text_probability_tolerance=1e-6,aligned_test_ids_and_labels=True,mask_records=len(maskrows),probe_records=len(probes),aggregate_script_sha256=sha_file(Path(__file__))))
 print("CORE_AGGREGATION_COMPLETE",len(sources),"checkpoints",len(maskrows),"mask records",flush=True)
def make_table(out,rows):
 idx={(r["dataset"],r["variant"],r["condition"]):r for r in rows}
 columns=("complete","T","TA","TV","p=0.1","p=0.3","p=0.5","p=0.7")
 lines=[r"\begin{table*}[t]",r"\centering",r"\caption{Core controls on the same non-neutral test sets and missing-modality masks. Macro-F1 (\%), mean $\pm$ sample standard deviation over five training seeds. At each missing rate, ten masks are averaged within each seed. Legacy text-only is independently trained with its historical fusion stack and auxiliary losses; it is not a plain BERT classifier.}",r"\label{tab:core_av_audit}",r"\resizebox{\textwidth}{!}{%",r"\begin{tabular}{llcccccccc}",r"\toprule",r"Dataset & Training variant & TAV & T & TA & TV & $p=0.1$ & $p=0.3$ & $p=0.5$ & $p=0.7$ \\",r"\midrule"]
 for d in DATASETS:
  for v in VARIANTS:
   vals=["$"+f'{100*idx[d,v,c]["macro_f1_mean"]:.2f} \\pm {100*idx[d,v,c]["macro_f1_sd"]:.2f}'+"$" for c in columns]
   label={"full":"TRIM (historical full)","no_modrop":"w/o modality dropout","text_only":"Legacy text-only"}[v]
   lines.append(d+" & "+label+" & "+" & ".join(vals)+r" \\")
  if d!=DATASETS[-1]:lines.append(r"\midrule")
 lines.extend([r"\bottomrule",r"\end{tabular}}",r"\end{table*}"])
 (out/"core_comparison_table.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=HERE/"results");aggregate(ap.parse_args())
