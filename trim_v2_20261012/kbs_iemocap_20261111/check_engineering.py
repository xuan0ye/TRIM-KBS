"""Synthetic correctness checks; no benchmark metrics are produced."""
import copy, tempfile
import numpy as np
import torch
from common import *
from data import fit_scaler, apply_scaler
from models import make_model
from interventions import *
from evaluate import encode_bank, predictor
from train import rng_state, restore_rng

def main():
    runtime('cpu', 17); checks = {}
    y = np.arange(6); perfect = np.eye(6) * 8; wrong = np.roll(perfect, 1, axis=1)
    assert metrics(y, perfect)['macro_f1'] == 1 and metrics(y, perfect)['uar'] == 1
    assert metrics(y, wrong)['macro_f1'] == 0 and metrics(y, wrong)['accuracy'] == 0
    assert np.isclose(metrics(y, np.zeros((6, 6)))['nll'], np.log(6))
    assert np.isclose(metrics(y, np.zeros((6, 6)))['brier'], 5/6)
    checks['six_class_metric_known_cases'] = True
    n = 6; aa = np.arange(n)[:, None] * np.arange(1, 7)[None, :]
    bb = np.linspace(.2, -.7, n)[:, None] * np.linspace(-1, 1, 6)[None, :]
    for anchor in [np.array([0, 2, 4, 5]), np.arange(n)]:
        full, half, _ = anchor_emap(n, anchor, lambda i, j: aa[i] + bb[j], pair_batch=5)
        assert np.max(np.abs(full - aa - bb)) < 1e-12 and np.max(np.abs(half - full)) < 1e-12
    sign = np.array([-1, -1, -1, 1, 1, 1]); direction = np.arange(6) - 2.5
    full, _, _ = anchor_emap(n, np.arange(n), lambda i, j: sign[i, None] * sign[j, None] * direction, pair_batch=5)
    assert np.max(np.abs(full)) < 1e-12
    checks['vector_emap_additive_noop_and_interaction_positive_control'] = True
    source = np.repeat(['a', 'b', 'c', 'd'], [3, 3, 2, 1])
    for seed in range(10):
        donor = cross_source(source, seed)
        assert np.all(source[donor] != source) and len(np.unique(donor)) == len(source)
        donor = within_source(source, seed); keep = donor >= 0
        assert keep.sum() == 8 and np.all(source[donor[keep]] == source[keep])
    checks['label_independent_bijective_donors_and_singleton_exclusion'] = True
    d = dict(audio=np.array([[1., 3.], [3., 7.], [0., 0.]], np.float32),
             vision=np.array([[2., 2.], [0., 0.], [6., 2.]], np.float32),
             observed=np.array([[1, 1], [1, 0], [0, 1]], bool))
    scaler = fit_scaler(d); changed = apply_scaler(copy.deepcopy(d), scaler)
    assert np.array_equal(scaler['audio_mean'], [2, 5])
    assert np.array_equal(changed['audio'][2], [0, 0]) and np.array_equal(changed['vision'][1], [0, 0])
    assert np.isfinite(changed['vision']).all()
    checks['fit_only_scaling_and_native_missing_rows'] = True
    t = torch.randint(0, 64, (6, 3, 8)); t[:, 1] = 1; t[:, 2] = 0
    a = torch.randn(6, 5); v = torch.randn(6, 7); observed = torch.ones((6, 2), dtype=torch.bool); observed[0, 1] = False
    identities = {}
    for variant in plan()['variants']:
        runtime('cpu', 23)
        model, base = make_model('tiny', 5, 7, variant, d_model=32, tiny=True, return_legacy=True)
        model.eval(); base.eval(); identities[variant] = model.identity()
        with torch.inference_mode():
            z, _ = model(t, a, v)
            if variant == 'trim':
                original, _ = base(t[:, 0], t[:, 1], t[:, 2], a, v, training=False)
                assert z.shape == (6, 6) and torch.allclose(z, original, atol=2e-6)
            off = torch.zeros((6, 2), dtype=torch.bool)
            x, _ = model(t, a, v, off); xx, _ = model(t, a * 100 + 4, v * 37 - 3, off)
            assert torch.allclose(x, xx, atol=1e-6)
            data = dict(y=np.arange(6), text=t.numpy(), audio=a.numpy(), vision=v.numpy(), observed=observed.numpy())
            bank = encode_bank(model, data, 'cpu', batch=3); fn = predictor(model, bank, 'cpu')
            ix = np.arange(6); donor = np.roll(ix, 1)
            for ai, vi in [(ix, ix), (donor, donor), (donor, ix), (ix, donor)]:
                mask = torch.stack([observed[ai, 0], observed[vi, 1]], 1)
                direct, _ = model(t, a[ai], v[vi], mask); direct = direct.numpy(); direct -= direct.mean(1, keepdims=True)
                assert np.max(np.abs(direct - fn(ix, ai, vi))) < 1e-5
        model.train(); z, aux = model(t, a, v, observed, auxiliary=True)
        loss = torch.nn.functional.cross_entropy(z, torch.arange(6)) + .02 * aux['nce'] + .02 * aux['ot']; loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        del model, base, bank
    assert len({identities[k]['common_branch_initialization_sha256'] for k in identities if k != 'bert_mlp'}) == 1
    assert abs(identities['budget_mlp']['fusion_budget']['relative_parameter_difference']) < .01
    checks['six_class_legacy_trim_forward_equivalence'] = True
    checks['five_variants_finite_backward_and_missing_invariance'] = True
    checks['cached_native_and_separate_modality_donors_match_forward'] = True
    checks['paired_initialization_and_nominal_fusion_parameter_budget'] = True
    runtime('cpu', 19); net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 6))
    opt = torch.optim.AdamW(net.parameters(), lr=.01); x = torch.randn(6, 4)
    def step():
        opt.zero_grad(); loss = (net(x) - torch.rand(6, 6)).square().mean(); loss.backward(); opt.step()
    step(); saved = dict(model=copy.deepcopy(net.state_dict()), optimizer=copy.deepcopy(opt.state_dict()), rng=rng_state())
    step(); expected = copy.deepcopy(net.state_dict())
    net.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng']); step()
    assert all(torch.equal(expected[k], net.state_dict()[k]) for k in expected)
    checks['stochastic_model_optimizer_resume_exact'] = True
    y = np.tile(np.arange(6), 2); source = np.repeat(np.arange(4).astype(str), 3)
    base = np.tile(y, (5, 1)); changed = base.copy(); changed[:, 0] = 1
    one = paired_bootstrap(base, changed, y, source, np.ones(12, bool), 64, 9)
    two = paired_bootstrap(base, changed, y, source, np.ones(12, bool), 64, 9)
    assert one == two and one['delta_macro_f1_pp'] < 0
    checks['paired_dialogue_and_seed_bootstrap_reproducible'] = True
    with tempfile.TemporaryDirectory() as tmp:
        try: evaluation_data(tmp, 'IEMOCAP', 'test')
        except FileNotFoundError: pass
        else: raise AssertionError('Unsealed evaluation was permitted')
    checks['unsealed_test_evaluation_rejected'] = True
    dump(HERE / 'checks/engineering_check.json', dict(status='passed', checks=checks, identities=identities,
         synthetic_engineering_only=True, formal_benchmark_training_jobs=0, formal_benchmark_evaluation_jobs=0))
    print('IEMO_ENGINEERING_PASSED', len(checks), flush=True)

if __name__ == '__main__': main()
