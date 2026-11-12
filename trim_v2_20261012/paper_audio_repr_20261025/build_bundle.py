"""Freeze source hashes and create a code-only upload ZIP."""
from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def build():
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.md','.json','.tex')
                 and not any(x in p.relative_to(HERE).parts for x in ('runs','checks_output','remote_results','feature_cache','.cache','__pycache__'))
                 and p.name!='bundle_manifest.json')
    manifest=dict(release='paper_audio_repr_20261025',revision='bert_config_equivalence_fix_20261025',files={p.relative_to(HERE).as_posix():sha(p) for p in files},
                  paper_model_source='audit_av_core_20261012/model_legacy.py',test_evaluated=False)
    path=HERE/'bundle_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    target=HERE.parent/(HERE.name+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[path]: z.write(p,HERE.name+'/'+p.relative_to(HERE).as_posix())
    print(target,'bytes',target.stat().st_size,'sha256',sha(target),'files',len(files)+1)
if __name__=='__main__': build()
