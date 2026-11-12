"""Real PANNs/media integration and compatibility check against the returned GPU probe."""
from pathlib import Path
import argparse,io,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_common import *
from paper_data import extract

def run(a):
    plan=read_json(a.audit/'inventory.json')
    with zipfile.ZipFile(a.probe) as z:
        name=next(n for n in z.namelist() if n.endswith('/events_train.npz'))
        with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as data: probe={k:data[k] for k in data.files}
    sid=str(probe['ids'][0]);row=next(r for r in plan['splits']['train'] if r['id']==sid)
    plan['splits']={'train':[row],'valid':[plan['splits']['valid'][0]]}
    a.out.mkdir(parents=True,exist_ok=True)
    extract(plan,a.out,a.assets,a.out/'cache','cpu')
    first=load_npz(a.out/'embedding_train.npz')['embedding'][0]
    delta=float(np.max(np.abs(first-probe['embedding'][0])))
    if delta>2e-4: raise RuntimeError(f'Paper extractor differs from the previous GPU probe: {delta}')
    manifest=read_json(a.out/'media_manifest.json')
    assert len(manifest['records'])==2
    for s in SPLITS:
        data=load_npz(a.out/f'embedding_{s}.npz')
        assert data['embedding'].shape==(1,2048) and np.isfinite(data['embedding']).all()
    # Exercise source-hash-verified cache reuse on both splits.
    extract(plan,a.out,a.assets,a.out/'cache','cpu')
    assert read_json(a.out/'media_manifest.json')==manifest
    report=dict(status='passed',real_raw_media=True,splits=['train','valid'],clips=2,device='cpu',
                previous_gpu_probe_max_embedding_delta=delta,cache_reuse=True,test_evaluated=False,
                extractor_sha256=sha_file(HERE/'panns_frozen.py'),checkpoint_sha256=PANN_SHA)
    dump_json(a.out/'real_extraction_check.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
