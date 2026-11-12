"""Temporary synthetic six-class run; all fake checkpoints/results are discarded."""
import copy, tempfile
import numpy as np
import common, train, evaluate, summarize, models
from common import *

def main():
    synthetic = copy.deepcopy(plan())
    synthetic.update(seeds=[11, 12], phases={'IEMOCAP': [1, 1]}, batch_size=4, accumulation_steps=2)
    synthetic['interventions']['repeats'] = 2; synthetic['bootstrap']['repeats'] = 32
    synthetic['emap'].update(anchors=4, banks=2)
    for module in [common, train, evaluate, summarize]: module.plan = lambda: synthetic
    original = models.make_model
    def tiny(bert, a_dim, v_dim, variant, **kwargs):
        kwargs.update(tiny=True, d_model=32); return original(bert, a_dim, v_dim, variant, **kwargs)
    train.make_model = tiny; evaluate.make_model = tiny
    rng = np.random.default_rng(8); y = np.tile(np.arange(6), 2).astype(np.int8)
    text = rng.integers(0, 64, (12, 3, 8), dtype=np.int64); text[:, 1] = 1; text[:, 2] = 0
    with tempfile.TemporaryDirectory(prefix='iemo_engineering_') as tmp:
        out = Path(tmp); folder = out / 'data/IEMOCAP'; folder.mkdir(parents=True)
        for role in ['fit', 'dev', 'test']:
            source = np.repeat([role + str(i) for i in range(4)], 3)
            d = dict(ids=np.array([s + '_' + str(i) for i, s in enumerate(source)]), source=source, y=y,
                     text=text, audio=rng.normal(size=(12, 5)).astype(np.float32), vision=rng.normal(size=(12, 7)).astype(np.float32),
                     observed=np.ones((12, 2), bool))
            d['observed'][0, 1] = False; d['vision'][0] = 0
            npz(folder / f'{role}.npz', **d)
        dump(folder / 'prepared.json', dict(signature={'synthetic_only': True}, files={x.name: sha(x) for x in folder.glob('*.npz')}))
        dump(out / 'bert_identity.json', {'synthetic_tiny_bert': True})
        for seed in synthetic['seeds']:
            for variant in synthetic['variants']: train.train_one(tmp, out, 'IEMOCAP', variant, seed, 'cpu')
        train.seal_training(out)
        for seed in synthetic['seeds']:
            for variant in synthetic['variants']: evaluate.evaluate_one(tmp, out, 'IEMOCAP', variant, seed, 'test', 'cpu')
        archive = summarize.summarize(out); assert archive.is_file()
        before = sha(out / 'training/IEMOCAP/trim/seed_11/best.pt')
        train.train_one(tmp, out, 'IEMOCAP', 'trim', 11, 'cpu')
        assert sha(out / 'training/IEMOCAP/trim/seed_11/best.pt') == before
        evaluate.evaluate_one(tmp, out, 'IEMOCAP', 'trim', 11, 'test', 'cpu')
        assert read(out / 'review/review_audit.json')['parameter_and_stochastic_pairing_verified']
    dump(HERE / 'checks/integration_check.json', dict(status='passed', synthetic_training_jobs=10, synthetic_evaluation_jobs=10,
         synthetic_epochs_per_job=2, temporary_predictions_and_weights_discarded=True, formal_benchmark_training_jobs=0,
         formal_benchmark_evaluation_jobs=0, cache_seal_and_stochastic_pairing_checked=True))
    print('IEMO_INTEGRATION_PASSED; synthetic artifacts discarded', flush=True)

if __name__ == '__main__': main()
