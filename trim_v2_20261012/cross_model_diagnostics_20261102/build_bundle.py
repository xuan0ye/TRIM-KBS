"""Build this source-only release; never includes trained weights or raw media."""
from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sources=[]
 references=json.loads((HERE/'reference_manifest.json').read_text('utf8'))
 for item in references:
  if sha(HERE/item['path'])!=item['sha256']:raise RuntimeError('Reference prediction changed: '+item['path'])
 groups=[('_trim',ROOT/'audit_av_core_20261012'),('_mult',ROOT/'src'),('CorMulT-main',ROOT/'CorMulT-main'),('_lnln',ROOT/'sota_repro_20261001/LNLN-main')]
 for prefix,original in groups:
  for f in (HERE/prefix).rglob('*'):
   if not f.is_file() or '__pycache__' in f.parts:continue
   origin=original/f.relative_to(HERE/prefix)
   if origin.is_file():
    a=sha(f);b=sha(origin)
    if a!=b:raise RuntimeError('Vendored inference source changed: '+str(f))
    sources.append(dict(vendored=f.relative_to(HERE).as_posix(),original=origin.relative_to(ROOT).as_posix(),sha256=a))
 (HERE/'source_provenance.json').write_text(json.dumps(sources,indent=2),encoding='utf8')
 exclude={'prepare_sources.py','bundle_manifest.json'}
 files=[p for p in HERE.rglob('*') if p.is_file() and not set(p.relative_to(HERE).parts)&{'__pycache__','runs','logs'} and not (len(p.relative_to(HERE).parts)==1 and (p.name in exclude or p.name.startswith('_')))]
 manifest=dict(release='cross-model-diagnostics-20261102',files={p.relative_to(HERE).as_posix():sha(p) for p in sorted(files)})
 target=HERE/'bundle_manifest.json';target.write_text(json.dumps(manifest,indent=2),encoding='utf8')
 archive=HERE.parent/(HERE.name+'.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in files+[target]:z.write(path,HERE.name+'/'+path.relative_to(HERE).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for rel,digest in manifest['files'].items():assert hashlib.sha256(z.read(HERE.name+'/'+rel)).hexdigest()==digest
 print(json.dumps(dict(zip=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(manifest['files']),vendored_sources_checked=len(sources))))
if __name__=='__main__':main()