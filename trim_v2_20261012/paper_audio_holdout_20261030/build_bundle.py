"""Immutable source-only held-out expert release."""
from pathlib import Path
import argparse,hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
PIN='7682f23806b0e309184bd9b2faa5ce011d1f9f08aeca1b232ecccc9f4e35b7c6'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 files=['holdout_split.py','holdout_core.py','holdout_router.py','run_holdout.py','split_plan.json','remote_audio_holdout.sh','PROTOCOL.md','REMOTE_README.md','build_bundle.py','checks/check_holdout.py']
 evidence=HERE/'checks/engineering_check.json'
 if not args.preflight:
  report=json.loads(evidence.read_text(encoding='utf8'))
  if report['status']!='pass':raise RuntimeError('Engineering checks failed')
  for f,digest in report['tested_source_sha256'].items():
   if sha(HERE/f)!=digest:raise RuntimeError('Code changed since check: '+f)
 if evidence.exists():files.append('checks/engineering_check.json')
 if sha(HERE.parent/'paper_audio_repr_20261025/bundle_manifest.json')!=PIN:raise RuntimeError('Paper source changed')
 manifest=dict(release='fresh_expert_video_holdout_20261030',parent_manifest_sha256=PIN,files={f:sha(HERE/f) for f in files},test_evaluated=False)
 (HERE/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 archive=HERE.parent/'paper_audio_holdout_20261030.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in files+['bundle_manifest.json']:z.write(HERE/f,HERE.name+'/'+f)
 print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(HERE/'bundle_manifest.json'))))
if __name__=='__main__':main()
