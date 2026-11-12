from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.md','.json')
                 and p.name!='bundle_manifest.json'
                 and not any(x in ('runs','logs','remote_results','__pycache__','.cache') for x in p.relative_to(HERE).parts))
    manifest=dict(release='paper_audio_causal_20261026',parent_manifest_sha256=sha(HERE.parent/'paper_audio_repr_20261025/bundle_manifest.json'),
                  files={p.relative_to(HERE).as_posix():sha(p) for p in files},test_evaluated=False)
    path=HERE/'bundle_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    archive=HERE.parent/(HERE.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[path]:z.write(p,HERE.name+'/'+p.relative_to(HERE).as_posix())
    print(archive,'bytes',archive.stat().st_size,'sha256',sha(archive))
if __name__=='__main__':build()
