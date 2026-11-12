"""Build an immutable, small remote source bundle without existing run artifacts."""
from pathlib import Path
import argparse,hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
PINS={'parent':('paper_audio_repr_20261025','7682f23806b0e309184bd9b2faa5ce011d1f9f08aeca1b232ecccc9f4e35b7c6'),
 'causal':('paper_audio_causal_20261026','dbd2bda89546e7b022414d61b8e5986d67de8d0e5fe8e6ff149555e5ee9d9875'),
 'information':('paper_audio_information_20261028','dd882e5e638041062e3b190b91687c67efe6efe5482eaa506524653964c3dc52')}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 files=['fusion_core.py','run_fusion.py','remote_panns_fusion.sh','PROTOCOL.md','REMOTE_README.md','build_bundle.py','checks/check_fusion.py']
 evidence=HERE/'checks/engineering_check.json'
 if not args.preflight and (not evidence.exists() or json.loads(evidence.read_text(encoding='utf8'))['status']!='pass'):
  raise RuntimeError('Run synthetic engineering checks before the release build')
 if evidence.exists():
  report=json.loads(evidence.read_text(encoding='utf8'))
  if not args.preflight:
   for rel,digest in report.get('tested_source_sha256',{}).items():
    if sha(HERE/rel)!=digest:raise RuntimeError('Source changed since the engineering check: '+rel)
   if len(report.get('tested_source_sha256',{}))!=3:raise RuntimeError('Missing tested-source provenance')
  files.append('checks/engineering_check.json')
 for name,(folder,digest) in PINS.items():
  if sha(HERE.parent/folder/'bundle_manifest.json')!=digest:raise RuntimeError('Dependency release changed: '+name)
 manifest=dict(release='supervised_full_PANNs_fusion_20261028',dependencies={k:v[1] for k,v in PINS.items()},
  files={f:sha(HERE/f) for f in files},test_evaluated=False)
 (HERE/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 archive=HERE.parent/'paper_audio_supervised_fusion_20261028.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in files+['bundle_manifest.json']:z.write(HERE/f,HERE.name+'/'+f)
 print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(HERE/'bundle_manifest.json'))))
if __name__=='__main__':main()
