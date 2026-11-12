"""Recompute all metrics and export a compact review without model weights."""
from pathlib import Path
import argparse
import shutil
import zipfile
import numpy as np
from event_common import *


def report(labels, logits):
    from aggregate_core import scores
    logits = np.asarray(logits,dtype=np.float64)
    probability = np.exp(logits-logits.max(-1,keepdims=True))
    probability /= probability.sum(-1,keepdims=True)
    return {name:scores(labels,logits[:,i].argmax(-1),probability[:,i,1])
            for i,name in enumerate(PATTERNS)}


def transition(y, before, after):
    a,b = before==y,after==y
    correction = int((~a & b).sum());harm = int((a & ~b).sum())
    return dict(wrong_to_right=correction,right_to_wrong=harm,net_corrections=correction-harm)


def summarize(out):
    out = Path(out)
    from event_train import VARIANTS, require_extraction
    extraction = require_extraction(out)
    protocol = read_json(out/"training_protocol.json")
    parent_record = read_json(out/"parent_cache_complete.json")
    for filename,digest in parent_record["files"].items():
        if sha_file(out/filename) != digest:
            raise RuntimeError(f"Parent cache changed: {filename}")
    parent = load_npz(out/"parent_valid.npz")
    labels, ids, base = parent["labels"],parent["ids"],parent["base_logits"]
    records = {"baseline":dict(logits=base)}
    selections = []
    for variant in VARIANTS:
        folder = out/"adapters"/variant/"seed_42"
        metadata = read_json(folder/"complete.json")
        if metadata["status"] != "complete" or metadata["test_evaluated"]:
            raise RuntimeError("Invalid adapter completion record.")
        if metadata["signature"]["variant"] != variant or metadata["signature"]["seed"] != 42:
            raise RuntimeError("Adapter identity differs.")
        for key,value in protocol.items():
            if metadata["signature"].get(key) != value:
                raise RuntimeError(f"Adapter protocol differs: {variant}/{key}")
        for name,digest in metadata["files"].items():
            if sha_file(folder/name) != digest:
                raise RuntimeError(f"Adapter artifact checksum differs: {folder/name}")
        values = load_npz(folder/"validation_predictions.npz")
        if not np.array_equal(values["ids"],ids) or not np.array_equal(values["labels"],labels):
            raise RuntimeError("Review prediction alignment failed.")
        if not np.array_equal(values["disabled_logits"],base):
            raise RuntimeError("Disabling event adapter must exactly restore baseline.")
        if not np.array_equal(values["logits"][:,(0,2)],base[:,(0,2)]):
            raise RuntimeError("Protected T/TV changed.")
        for permutation in values["shuffle_indices"]:
            if sorted(permutation.tolist()) != list(range(len(ids))) or np.any(permutation==np.arange(len(ids))):
                raise RuntimeError("Invalid event shuffle.")
        calculated = report(labels,values["logits"])
        score = float(np.mean([r["macro_f1"] for r in calculated.values()]))
        selected = read_json(folder/"best_validation.json")
        if abs(score-selected["selection_score"]) > 1e-12:
            raise RuntimeError("Stored selected score differs.")
        selections.append(dict(variant=variant,seed=42,selected_epoch=metadata["selected_epoch"],
                               selected_epoch_zero=metadata["selected_epoch_zero"],
                               training_gain_pp=100*(score-metadata["initial_selection_score"]),
                               trainable_parameters=metadata["trainable_parameters"]))
        records[variant] = values
    if len({r["trainable_parameters"] for r in selections}) != 1:
        raise RuntimeError("Adapter capacities differ across variants.")
    baseline_scores = report(labels,base)
    metrics, paired, additions, interventions, comparison = [], [], [], [], []
    for variant, values in records.items():
        measured = report(labels,values["logits"])
        for i,name in enumerate(PATTERNS):
            metrics.append(dict(dataset="MOSEI",variant=variant,seed=42,condition=name,**measured[name]))
            paired.append(dict(variant=variant,condition=name,
                          macro_f1_delta_pp=100*(measured[name]["macro_f1"]-baseline_scores[name]["macro_f1"]),
                          **transition(labels,base[:,i].argmax(-1),values["logits"][:,i].argmax(-1))))
        for prior,new in ((0,1),(2,3)):
            additions.append(dict(variant=variant,comparison=PATTERNS[new]+" - "+PATTERNS[prior],
                             macro_f1_delta_pp=100*(measured[PATTERNS[new]]["macro_f1"]-measured[PATTERNS[prior]]["macro_f1"]),
                             **transition(labels,values["logits"][:,prior].argmax(-1),values["logits"][:,new].argmax(-1))))
        average = float(np.mean([m["macro_f1"] for m in measured.values()]))
        comparison.append(dict(variant=variant,seed=42,mean_macro_f1_pct=100*average,
                          delta_vs_baseline_pp=100*(average-np.mean([m["macro_f1"] for m in baseline_scores.values()]))))
        if "shuffle_logits" in values:
            for k,logits in enumerate(values["shuffle_logits"]):
                shifted = report(labels,logits)
                for name in PATTERNS:
                    interventions.append(dict(variant=variant,shuffle_seed=(13,21,42)[k],condition=name,
                        macro_f1_drop_pp=100*(measured[name]["macro_f1"]-shifted[name]["macro_f1"])))
    event = load_npz(out/"events_valid.npz")
    if not np.array_equal(event["ids"],ids):
        raise RuntimeError("Validation event IDs differ.")
    watch = [i for i,name in enumerate(event["labels"]) if str(name) in
             ("Explosion","Gunshot, gunfire","Fireworks","Screaming","Crying, sobbing","Laughter","Sigh","Applause","Siren")]
    subset = (event["events"][:,watch]>=.5).any(1) if watch else np.zeros(len(ids),bool)
    subset_rows = []
    for variant, values in records.items():
        for name,selected in (("watched_event_score_ge_0.5",subset),("other",~subset)):
            if selected.any():
                for condition,metric in report(labels[selected],values["logits"][selected]).items():
                    subset_rows.append(dict(variant=variant,subset=name,condition=condition,**metric))
    summary = out/"summary"
    summary.mkdir(exist_ok=True)
    for name, rows in (("validation_metrics",metrics),("paired_error_changes",paired),
                       ("audio_addition",additions),("event_interventions",interventions),
                       ("selection",selections),("event_subsets",subset_rows)):
        write_csv(summary/(name+".csv"),rows)
    result = dict(status="complete",dataset="MOSEI",scope="validation_only",smoke=protocol["smoke"],
                  n_seeds=1,test_evaluated=False,comparisons=comparison,
                  selection=selections,protected_T_TV_exact=True,
                  note="Single seed screening. PANNs event scores are not fear labels. Fixed projections equalize adapter capacity but introduce representation/projection differences; no novelty or causal claim.")
    dump_json(summary/"event_comparisons.json",result)
    atomic_npz(summary/"baseline_validation.npz",ids=ids,labels=labels,logits=base)
    atomic_npz(summary/"validation_event_scores.npz",ids=ids,events=event["events"],
               event_labels=event["labels"],watched_event_subset=subset)
    review = out/"review"
    review.mkdir(exist_ok=True)
    names = ("audit.json","inventory.json","extraction_signature.json","extraction_complete.json",
             "extraction_progress.json","event_coverage_train.csv","listening_sample_train.csv",
             "media_manifest.json","parent_source.json","parent_cache_complete.json",
             "training_protocol.json","training_progress.json","training_shuffle_indices.npz")
    for name in names:
        shutil.copy2(out/name,review/name)
    for path in summary.iterdir():
        if path.is_file():
            target = review/"summary"/path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path,target)
    for variant in VARIANTS:
        source = out/"adapters"/variant/"seed_42"
        target = review/"adapters"/variant/"seed_42"
        target.mkdir(parents=True,exist_ok=True)
        for name in ("complete.json","best_validation.json","history.csv","validation_predictions.npz"):
            shutil.copy2(source/name,target/name)
    if (HERE/"bundle_manifest.json").is_file():
        shutil.copy2(HERE/"bundle_manifest.json",review/"code_bundle_manifest.json")
    files = {p.relative_to(review).as_posix():sha_file(p) for p in review.rglob("*")
             if p.is_file() and p.name!="review_manifest.json"}
    dump_json(review/"review_manifest.json",dict(release=RELEASE,files=files,test_evaluated=False,
              contains_model_weights=False,scope="validation_only",smoke=protocol["smoke"]))
    archive = out/f"event_review_{'smoke' if protocol['smoke'] else 'pilot'}_20261024.zip"
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(review.rglob("*")):
            if path.is_file():
                zipped.write(path,path.relative_to(review.parent).as_posix())
    temporary.replace(archive)
    print("EVENT_REVIEW_COMPLETE",archive,"test_evaluated",False,flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    summarize(args.out)


if __name__ == "__main__":
    main()
