import importlib
import json

import pytest
import torch


def leap():
    try:
        return importlib.import_module("leap")
    except ModuleNotFoundError as error:
        if error.name != "leap":
            raise
        pytest.fail("LEAP core has not been implemented")


def grid(n):
    return torch.linspace(1, 0, n + 1, dtype=torch.float64)


def test_prediction_indices_end_in_true_evaluation():
    assert leap().prediction_steps(25) == tuple(range(5, 24, 2))
    assert leap().prediction_steps(6) == ()
    assert leap().prediction_steps(8) == (5,)


def test_ridge_matches_direct_stacked_fit_and_partition_merge():
    m = leap()
    generator = torch.Generator().manual_seed(17)
    refs = [torch.randn(7, 1, 4, generator=generator, dtype=torch.float64) for _ in range(3)]
    kwargs = dict(latent_shape=(1, 4), metadata={"cfg": 2.5}, window=3, ridge=.2)
    full, left, right = [m.LeapAccumulator(grid(7), **kwargs) for _ in range(3)]
    for i, ref in enumerate(refs):
        full.add_trajectory(ref)
        (left if i == 0 else right).add_trajectory(ref)
    left.merge(right)
    bank = full.finish()
    assert bank.reference_count == 3
    for i, row in bank.coefficients.items():
        # Oracle spells out direction and order; flatten independently of solver.
        bs = torch.cat([torch.stack((v[i-3]-v[i-2], v[i-2]-v[i-1], v[i-1]), -1).reshape(-1, 3) for v in refs])
        y = torch.cat([v[i].flatten() for v in refs])
        A = torch.cat((bs, .2**.5 * torch.eye(3, dtype=bs.dtype)))
        z = torch.cat((y, torch.zeros(3, dtype=y.dtype)))
        expect = torch.linalg.lstsq(A, z).solution
        assert torch.allclose(torch.tensor(row, dtype=expect.dtype), expect, rtol=1e-12, atol=1e-12)
        assert left.finish().coefficients[i] == pytest.approx(row, rel=1e-12, abs=1e-12)
    loaded = m.LeapAccumulator.from_dict(json.loads(json.dumps(left.to_dict())))
    loaded_bank = m.LeapBank.from_dict(json.loads(json.dumps(bank.to_dict())))
    assert loaded.finish().coefficients == left.finish().coefficients
    assert loaded_bank.to_dict() == bank.to_dict()
    loaded_bank.validate(grid(7), (1, 4), {"cfg": 2.5})


def test_incompatible_calibration_fails_without_leaving_partial_statistics():
    m = leap()
    acc = m.LeapAccumulator(grid(9), latent_shape=(1, 2), metadata={"cfg": 2.5})
    good = torch.ones(9, 1, 2, dtype=torch.float64)
    acc.add_trajectory(good)
    original = acc.to_dict()
    bad = good.clone()
    bad[-1] = float("nan")
    for item in (bad, good[:-1], list(good) + [good[-1]], good[:, :, :1]):
        with pytest.raises(ValueError):
            acc.add_trajectory(item)
        assert acc.to_dict() == original
    with pytest.raises(ValueError, match="metadata"):
        acc.finish().validate(grid(9), (1, 2), {"cfg": 5})
    with pytest.raises(ValueError, match="sigma"):
        acc.finish().validate(grid(9) ** 2, (1, 2), {"cfg": 2.5})
    with pytest.raises(ValueError, match="shape"):
        acc.finish().validate(grid(9), (1, 3), {"cfg": 2.5})
    bad_record = acc.finish().to_dict()
    bad_record["reference_count"] = False
    with pytest.raises(ValueError):
        m.LeapBank.from_dict(bad_record)


def test_inference_uses_prediction_history_and_exact_call_at_reached_state():
    m = leap()
    sigmas = grid(25)
    shape = (1, 2)
    x0 = torch.tensor([[.8, -1.2]], dtype=torch.float64)
    acc = m.LeapAccumulator(sigmas, latent_shape=shape, metadata={})
    acc.add_trajectory(torch.ones(25, *shape, dtype=x0.dtype))
    record = acc.finish().to_dict()
    # Vary coefficients explicitly to distinguish replacing predicted history with exact.
    custom = [(.13, -.42, .19, .31, .87) if i % 4 == 1 else (-.12, .18, .07, -.93, .64) for i in range(5, 24, 2)]
    record["coefficients"] = {str(i): list(row) for i, row in zip(range(5, 24, 2), custom)}
    bank = m.LeapBank.from_dict(record)
    calls, snapshots = [], []
    bias = torch.tensor([[.5, -.3]], dtype=x0.dtype)

    def actual_model(x, sigma, **args):
        assert args == {"seed": 1337}
        calls.append((float(sigma[0]), x.clone()))
        return x - sigma[0] * (.33 * x + sigma[0] * bias)

    result = m.sample_leap_euler(actual_model, x0, sigmas, {"seed": 1337}, snapshots.append, bank=bank)
    expect_calls = []
    x = x0.clone()
    history = []
    for i in range(25):
        s = sigmas[i]
        if i in range(5, 24, 2):
            a, b, c, d, e = custom[(i - 5)//2]
            h0, h1, h2, h3, h4 = history[-5:]
            velocity = a*(h0-h1) + b*(h1-h2) + c*(h2-h3) + d*(h3-h4) + e*h4
        else:
            expect_calls.append((float(s), x.clone()))
            velocity = .33*x + s*bias
        assert snapshots[i]["i"] == i
        assert torch.equal(snapshots[i]["sigma"], s)
        assert torch.equal(snapshots[i]["sigma_hat"], s)
        assert torch.allclose(snapshots[i]["denoised"], x-s*velocity, atol=1e-14, rtol=0)
        history.append(velocity)
        x = x + velocity * (sigmas[i+1]-s)
    assert len(calls) == len(expect_calls) == 15
    assert calls[-1][0] == float(sigmas[-2])
    for (actual_s, actual_x), (expected_s, expected_x) in zip(calls, expect_calls):
        assert actual_s == expected_s
        assert torch.allclose(actual_x, expected_x, atol=1e-14, rtol=0)
    assert torch.allclose(x, result, atol=1e-14, rtol=0)
    assert torch.equal(x0, torch.tensor([[.8, -1.2]], dtype=x0.dtype))


def test_reference_euler_observer_and_reject_nonzero_final_before_call():
    m = leap()
    visited = []

    def model(x, sigma):
        return x - sigma[:, None] * torch.tensor([.5, -.25])

    x = torch.zeros(2, 2)
    sigmas = grid(4).to(x.dtype)
    out = m.sample_euler_reference(model, x, sigmas, on_velocity=lambda i, velocity: visited.append((i, velocity)))
    assert [i for i, _ in visited] == list(range(4))
    assert torch.equal(out, torch.tensor([[-.5, .25], [-.5, .25]]))
    with pytest.raises(ValueError, match="sigma"):
        m.sample_euler_reference(lambda *_: pytest.fail("must check grid first"), x, sigmas + .1)


@pytest.mark.parametrize("failure", ["missing", "extra", "nan", "negative_gram"])
def test_cannot_load_an_incomplete_or_invalid_saved_calibration(failure):
    m = leap()
    acc = m.LeapAccumulator(grid(8), latent_shape=(1, 2), metadata={})
    acc.add_trajectory(torch.ones(8, 1, 2))
    if failure == "negative_gram":
        record = acc.to_dict()
        record["gram"][0][0][0] = -1.0
        with pytest.raises(ValueError, match="statistics"):
            m.LeapAccumulator.from_dict(record)
    else:
        record = acc.finish().to_dict()
        if failure == "missing":
            del record["coefficients"]["5"]
        elif failure == "extra":
            record["coefficients"]["05"] = record["coefficients"]["5"]
        else:
            record["coefficients"]["5"][0] = float("nan")
        with pytest.raises(ValueError, match="coefficient"):
            m.LeapBank.from_dict(record)


def test_large_random_trajectory_statistics_can_reload_their_own_record():
    m = leap()
    shape = (1, 128, 32, 32)
    g = torch.Generator().manual_seed(5)
    acc = m.LeapAccumulator(grid(8), latent_shape=shape, metadata={})
    acc.add_trajectory(torch.randn(8, *shape, generator=g))
    canonical = acc.to_dict()
    assert m.LeapAccumulator.from_dict(canonical).to_dict() == canonical
    # Older saved data may differ by one FP64 rounding unit across the diagonal.
    perturbed = json.loads(json.dumps(canonical))
    row = perturbed["gram"][0][0]
    row[1] = float(torch.nextafter(torch.tensor(row[1], dtype=torch.float64), torch.tensor(float("inf"), dtype=torch.float64)))
    assert m.LeapAccumulator.from_dict(perturbed).finish().coefficients[5] == pytest.approx(
        acc.finish().coefficients[5], abs=1e-12
    )


def test_extreme_finite_coefficients_and_impossible_statistics_are_rejected():
    m = leap()
    acc = m.LeapAccumulator(grid(8), latent_shape=(1, 2), metadata={})
    acc.add_trajectory(torch.zeros(8, 1, 2))
    bank = acc.finish().to_dict()
    bank["coefficients"]["5"][0] = 1e300
    with pytest.raises(ValueError, match="coefficient"):
        m.LeapBank.from_dict(bank)
    malformed = acc.to_dict()
    malformed["rhs"][0][0] = 1e200
    with pytest.raises(ValueError, match="statistics"):
        m.LeapAccumulator.from_dict(malformed)


def test_fit_refuses_arithmetically_excessive_gain_without_silently_clipping():
    m = leap()
    acc = m.LeapAccumulator(grid(8), latent_shape=(1, 2), metadata={})
    history = torch.ones(8, 1, 2)
    history[5] *= 1e6
    acc.add_trajectory(history)
    with pytest.raises(ValueError, match="coefficient"):
        acc.finish()
