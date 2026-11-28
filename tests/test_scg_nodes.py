import importlib
import json
import runpy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


class NativeDiffusion:
    pass


class Flow:
    multiplier = 1.0
    shift = 5.0
    sigma_max = 1.0
    noise_scale = 0.0


class CONST:
    def calculate_denoised(self, sigma, output, x):
        return x - sigma * output

    def noise_scaling(self, sigma, noise, latent, max_denoise=False):
        return sigma * self.noise_scale * noise + (1 - sigma) * latent


class Sampling(Flow, CONST):
    pass


class ModelPatcher:
    def __init__(self):
        def process(latent):
            self.normalized += 1
            return latent * 2

        self.normalized = 0
        self.model = SimpleNamespace(
            diffusion_model=NativeDiffusion(), model_sampling=Sampling(), process_latent_in=process,
        )
        self.model_options = {}
        self.patches = {}

    def get_model_object(self, name):
        return getattr(self.model, name)


class VAE:
    latent_channels = 128
    downscale_ratio = 16

    def __init__(self, latent):
        self.latent = latent
        self.inputs = []

    def encode(self, images):
        self.inputs.append(images)
        return self.latent


class CFGGuider:
    def __init__(self, patcher):
        self.model_patcher = patcher
        self.inner_model = patcher.model
        self.conds = {"positive": [{}], "negative": [{}]}

    def predict_noise(self, x, *args, **kwargs):
        return x


class KSamplerX0Inpaint:
    def __init__(self, model):
        self.inner_model = CFGGuider(model)
        self.latent_image = None
        self.calls = []

    def __call__(self, x, sigma, **kwargs):
        self.calls.append((x, sigma))
        # A small nonzero derivative makes callback and Euler order observable.
        return x - sigma.view(-1, 1, 1, 1) * 0.25 * x


def stock_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    for i, (sigma, next_sigma) in enumerate(zip(sigmas, sigmas[1:])):
        denoised = model(x, sigma.expand(x.shape[0]), **(extra_args or {}))
        d = (x - denoised) / sigma
        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        x = x + d * (next_sigma - sigma)
    return x


@pytest.fixture
def nodes(monkeypatch):
    def register(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    def port(name=None, **kw):
        return SimpleNamespace(name=name, **kw)

    ports = SimpleNamespace(Input=port, Output=port)
    io = SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}), Schema=lambda **kw: SimpleNamespace(**kw),
        NodeOutput=lambda *items: SimpleNamespace(result=items),
        Model=ports, Vae=ports, Image=ports, Float=ports, Combo=ports, String=ports, Sampler=ports, Custom=lambda _: ports,
    )
    ksampler = lambda name: stock_euler if name == "euler" else None
    samplers = register("comfy.samplers", KSAMPLER=lambda fn: fn, ksampler=ksampler,
                        KSamplerX0Inpaint=KSamplerX0Inpaint, CFGGuider=CFGGuider)
    k_diff = register("comfy.k_diffusion", __path__=[])
    k_diff.sampling = register("comfy.k_diffusion.sampling", sample_euler=stock_euler,
                              to_d=lambda x, sigma, denoised: (x - denoised) / sigma)
    register("comfy", __path__=[], samplers=samplers, k_diffusion=k_diff)
    for name in ("comfy.ldm", "comfy.ldm.moss_image", "comfy_api"):
        register(name, __path__=[])
    register("comfy_api.latest", io=io)
    register("comfy.ldm.moss_image.model", MOSSImage=NativeDiffusion)
    register("comfy.model_patcher", ModelPatcher=ModelPatcher)
    register("comfy.model_sampling", CONST=CONST, ModelSamplingDiscreteFlow=Flow)
    register("comfy.sd", VAE=VAE)
    package = "_moss_scg_nodes_test"
    register(package, __path__=[str(Path(__file__).resolve().parents[1])])
    try:
        yield importlib.import_module(package + ".scg_nodes")
    except ModuleNotFoundError as error:
        if error.name == package + ".scg_nodes":
            pytest.fail("SCG ComfyUI adapter is not implemented")
        raise
    finally:
        for name in (package + ".scg", package + ".scg_nodes"):
            sys.modules.pop(name, None)


@pytest.fixture
def inputs():
    plane = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0]).repeat(3, 1)
    x = plane.expand(1, 128, 3, 5).clone()
    return ModelPatcher(), VAE(x / 2), torch.zeros(1, 48, 80, 3), x


def build(nodes, inputs):
    model, vae, images, _ = inputs
    return nodes.MossImageSCGProfile.execute(model, vae, images).result[0]


def sampler(nodes, profile=None, **params):
    return nodes.MossImageSCGEuler.execute(profile=profile, **params).result[0]


@pytest.mark.parametrize("stage,grid", [("first_pass", [1., .7, .4]), ("second_pass", [.8, .4, 0.])])
def test_scg_pass_uses_only_supplied_updates(nodes, inputs, stage, grid):
    model, _, _, x = inputs
    wrapped = KSamplerX0Inpaint(model)
    if stage == "second_pass":
        wrapped.latent_image = x * .3
    sigmas = torch.tensor(grid)
    actual = sampler(nodes, sampling_pass=stage)(wrapped, x, sigmas, disable=True)
    assert [s.item() for _, s in wrapped.calls] == pytest.approx(grid[:-1])
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, stock_euler(KSamplerX0Inpaint(model), x, sigmas))


@pytest.mark.parametrize("stage,grid", [
    ("first_pass", [.8, .5]), ("second_pass", [.8, .5]), ("second_pass", [1., .5, 0.]),
    ("second_pass", [.8, .8, 0.]), ("first_pass", [1., float("nan"), .4]),
])
def test_scg_pass_rejects_invalid_grid_before_sampling(nodes, inputs, stage, grid):
    model, _, _, x = inputs
    wrapped = KSamplerX0Inpaint(model)
    with pytest.raises(ValueError, match="sigma"):
        sampler(nodes, sampling_pass=stage)(wrapped, x, torch.tensor(grid), disable=True)
    assert not wrapped.calls


def test_schema_and_calibration_use_input_vae_and_normalize_once(nodes, inputs):
    model, vae, images, x = inputs
    profile_node = nodes.MossImageSCGProfile.define_schema()
    sampler_node = nodes.MossImageSCGEuler.define_schema()
    assert [p.name for p in profile_node.inputs] == ["model", "vae", "images"]
    assert [p.name for p in sampler_node.inputs] == ["profile", "strength", "sigma_min", "sigma_max", "debug_log",
                                                      "within_bin_mix", "budget_trace", "spectrum_space", "sampling_pass"]
    assert [p.default for p in sampler_node.inputs[-5:]] == ["off", 0., "", "packed", "full"]
    profile = build(nodes, inputs)
    assert model.normalized == 1
    assert vae.inputs == [images]
    assert profile.latent_shape == (128, 3, 5)
    assert profile.image_shape == (48, 80)
    # Parseval for orthonormal FFT: weighted radial bin powers sum to spatial energy.
    from _moss_scg_nodes_test.scg import radial_geometry
    counts = radial_geometry(3, 5).counts
    torch.testing.assert_close((profile.power * counts).sum(), x[0, 0].square().sum())


def test_paired_spatial_comparison_records_packed_snapshots_and_matches_total(nodes, inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    import numpy as np
    model, _, _, original = inputs
    profile = build(nodes, inputs)
    x = torch.randn(original.shape, generator=torch.Generator().manual_seed(87))
    sigmas = torch.tensor([1., .8, .5, 0.])
    def run(space, control=""):
        loop = sampler(nodes, profile, strength=.15, spectrum_space=space,
                       budget_trace=control, debug_log="jsonl+latents")
        return loop(KSamplerX0Inpaint(model), x, sigmas, disable=True)
    packed = run("packed_measured")
    root = tmp_path / "moss_scg_debug"
    base, = root.iterdir()
    spatial = run("spatial_measured", base.name)
    new, = [d for d in root.iterdir() if d != base]
    assert not torch.allclose(spatial, packed)
    records = [json.loads(line) for line in (new / "trace.jsonl").read_text().splitlines()]
    assert records[0]["spectral_shape"] == [32,6,10]
    assert records[0]["budget_mode"] == "total_l2" and records[0]["shape"] == list(x.shape)
    with np.load(new / "step_0001.npz") as states:
        assert states["after"].shape == tuple(x.shape)
    compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/compare_scg_traces.py"))["compare"]
    rows = list(compare(base, new))
    assert all(abs(r["scg_l2_ratio"]-1)<1e-4 for r in rows[:-1])
    assert rows[0]["batch_denoised_identical"]
    header, *tail = (new / "trace.jsonl").read_text().splitlines()
    altered_header = json.loads(header)
    altered_header["paired_reference_id"] = "different calibration"
    (new / "trace.jsonl").write_text("\n".join([json.dumps(altered_header), *tail]))
    with pytest.raises(ValueError, match="paired reference"):
        list(compare(base, new))


def test_bundled_spaces_require_native_calibration_shape_and_match_pair(nodes, inputs):
    from _moss_scg_nodes_test.scg import radial_geometry
    packed = torch.empty(1,128,64,64)
    a, key_a, count_a = nodes._reference_for("packed_measured", None, packed, radial_geometry(64,64))
    b, key_b, count_b = nodes._reference_for("spatial_measured", None, packed, radial_geometry(128,128))
    assert key_a == key_b and count_a == count_b == 9
    assert len(a) == 46 and len(b) == 92 and torch.all(a > 0) and torch.all(b > 0)
    with pytest.raises(ValueError, match="1024x1024"):
        nodes._reference_for("spatial_measured", None, inputs[-1], radial_geometry(6,10))


def test_scg_changes_only_post_update_state_for_next_nonterminal_sigma(nodes, inputs):
    model, _, _, x = inputs
    loop = sampler(nodes, build(nodes, inputs), strength=1, sigma_min=.5, sigma_max=.5)
    wrapped = KSamplerX0Inpaint(model)
    events = []
    out = loop(wrapped, x, torch.tensor([1.0, .5, 0.0]), callback=events.append, disable=True)
    # The first Euler step is 0.875x. Target power at next sigma=.5 is
    # 0.25 of the original, yielding 0.5x; terminal Euler adds no SCG.
    assert [sigma.item() for _, sigma in wrapped.calls] == [1.0, .5]
    assert wrapped.calls[0][0] is x
    torch.testing.assert_close(wrapped.calls[1][0], x * .5, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(events[1]["x"], x * .5, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, x * .4375, atol=1e-5, rtol=1e-5)


def test_zero_and_inactive_window_skip_fft_and_preserve_euler(nodes, inputs, monkeypatch):
    model, _, _, x = inputs
    zero = sampler(nodes, strength=0)
    assert zero is stock_euler
    disabled_window = sampler(nodes, build(nodes, inputs), strength=1, sigma_min=.6, sigma_max=.9)
    monkeypatch.setattr(torch.fft, "fft2", lambda *a, **k: pytest.fail("inactive path ran FFT"))
    sigmas = torch.tensor([1.0, .5, 0.0])
    expected = stock_euler(KSamplerX0Inpaint(model), x, sigmas)
    actual = disabled_window(KSamplerX0Inpaint(model), x, sigmas, disable=True)
    assert torch.equal(actual, expected)


def test_disabled_sampler_does_not_request_a_connected_profile(nodes, inputs):
    node = nodes.MossImageSCGEuler
    assert node.define_schema().inputs[0].lazy is True
    assert node.check_lazy_status(strength=0.0) == []
    assert node.check_lazy_status(strength=0.15) == []
    assert node.check_lazy_status(profile=None, strength=0.15) == ["profile"]
    assert node.check_lazy_status(profile=None, strength=0, debug_log="jsonl") == ["profile"]
    assert node.check_lazy_status(profile=build(nodes, inputs), strength=0.15) == []


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda w, x, s, kw: kw.update(denoise_mask=torch.ones_like(x)), "mask"),
        (lambda w, x, s, kw: setattr(w, "latent_image", x), "T2I"),
        (lambda w, x, s, kw: s.__setitem__(0, .75), "full"),
        (lambda w, x, s, kw: s.__setitem__(-1, .25), "full"),
        (lambda w, x, s, kw: setattr(w.inner_model, "model_patcher", ModelPatcher()), "MODEL instance"),
        (lambda w, x, s, kw: w.inner_model.conds["positive"][0].update(model_conds={"ref_latents": [x]}), "reference"),
        (lambda w, x, s, kw: w.inner_model.conds["positive"][0].update(model_conds={"ref_kv": object()}), "reference"),
        (lambda w, x, s, kw: w.inner_model.conds["positive"][0].update(concat_latent_image=x), "reference"),
        (lambda w, x, s, kw: kw.update(model_options={"sampler_post_cfg_function": [lambda a: a]}), "sampler_post_cfg_function"),
    ],
)
def test_unsupported_sampler_data_rejects_before_a_denoiser_call(nodes, inputs, mutate, match):
    model, _, _, x = inputs
    loop = sampler(nodes, build(nodes, inputs))
    wrapped, sigmas, extra = KSamplerX0Inpaint(model), torch.tensor([1.0, .5, 0.0]), {}
    mutate(wrapped, x, sigmas, extra)
    with pytest.raises(ValueError, match=match):
        loop(wrapped, x, sigmas, extra_args=extra, disable=True)
    assert not wrapped.calls


def test_bad_profile_shape_changed_sampling_and_invalid_profile_are_explicit(nodes, inputs):
    model, _, _, x = inputs
    loop = sampler(nodes, build(nodes, inputs))
    wrapped, sigmas = KSamplerX0Inpaint(model), torch.tensor([1.0, .5, 0.0])
    with pytest.raises(ValueError, match="shape"):
        loop(wrapped, x[..., :-1], sigmas, disable=True)
    model.model.model_sampling.shift = 4.0
    with pytest.raises(ValueError, match="sampling settings"):
        loop(wrapped, x, sigmas, disable=True)
    with pytest.raises(ValueError, match="profile"):
        sampler(nodes, profile=object(), strength=.1)


def test_text_only_scg_uses_nonzero_correction(nodes, inputs):
    model, _, _, x = inputs
    sigmas = torch.tensor([1., .5, 0.])
    loop = sampler(nodes, strength=.15)
    wrapped = KSamplerX0Inpaint(model)
    result = loop(wrapped, x, sigmas, disable=True)
    assert len(wrapped.calls) == 2
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, stock_euler(KSamplerX0Inpaint(model), x, sigmas))
    with pytest.raises(ValueError, match="128"):
        loop(wrapped, x[:, :1], sigmas, disable=True)


@pytest.mark.parametrize("measured_profile", [False, True])
def test_scg_accepts_diffusion_weight_patches_and_static_row_scaling(nodes, inputs, measured_profile):
    model, _, _, x = inputs
    weight = "diffusion_model.layers.0.proj.qkv.weight"
    model.patches = {weight: [object()]}
    model.weight_wrapper_patches = {weight: [lambda w: w * .9]}
    model.model_options["transformer_options"] = {"moss_block_tune_preset": "image_3_individual_qkv_partial"}
    profile = build(nodes, inputs) if measured_profile else None
    result = sampler(nodes, profile)(KSamplerX0Inpaint(model), x, torch.tensor([1., .5, 0.]), disable=True)
    assert torch.isfinite(result).all()


def test_calibration_rejects_non_native_and_uncropped_reference(nodes, inputs):
    model, vae, images, _ = inputs
    with pytest.raises(ValueError, match="divisible"):
        nodes.MossImageSCGProfile.execute(model, vae, images[:, :-1])
    assert not vae.inputs
    model.model.model_sampling.multiplier = 1000
    with pytest.raises(ValueError, match="multiplier"):
        build(nodes, inputs)


@pytest.mark.parametrize("name", ("wrappers", "weight_wrapper_patches", "injections", "additional_models", "callbacks"))
def test_patcher_extensions_fail_before_reference_encoding(nodes, inputs, name):
    model, vae, _, _ = inputs
    setattr(model, name, {"registered": [object()]})
    with pytest.raises(ValueError, match=name):
        build(nodes, inputs)
    assert not vae.inputs


def test_non_sampling_object_patch_is_rejected(nodes, inputs):
    model, _, _, _ = inputs
    model.object_patches = {"diffusion_model.video_embedder": object()}
    with pytest.raises(ValueError, match="object patch"):
        build(nodes, inputs)


@pytest.mark.parametrize("strength", [0, 1])
def test_debug_records_actual_euler_updates_and_preserves_output(nodes, inputs, tmp_path, monkeypatch, strength):
    import numpy as np
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    model, _, _, x = inputs
    profile = build(nodes, inputs)
    x = torch.cat((x, 2*x))
    sigmas = torch.tensor([1., .75, .5, 0.])
    kwargs = dict(strength=strength, sigma_min=.5, sigma_max=.5)
    expected = sampler(nodes, profile, **kwargs)(KSamplerX0Inpaint(model), x, sigmas, disable=True)
    assert not list(tmp_path.iterdir())
    for mode in ("jsonl", "jsonl+latents"):
        wrapped = KSamplerX0Inpaint(model)
        output = sampler(nodes, profile, debug_log=mode, **kwargs)(
            wrapped, x, sigmas, extra_args={"seed": 42}, disable=True)
        assert torch.equal(output, expected) and len(wrapped.calls) == 3
    runs = list((tmp_path / "moss_scg_debug").glob("*/trace.jsonl"))
    assert len(runs) == 2
    for path in runs:
        header, first, second, last, end = [json.loads(s) for s in path.read_text().splitlines()]
        assert header["seed"] == 42 and header["reference_kind"] == "measured"
        assert header["sigmas"] == sigmas.tolist() and header["shape"] == list(x.shape)
        assert end["status"] == "complete" and end["steps_written"] == 3
        assert [a["step"] for a in (first, second, last)] == [1, 2, 3]
        assert first["reason"] == ("outside_window" if strength else "strength_zero")
        assert last["reason"] == "terminal"
        assert second["reason"] == ("applied" if strength else "strength_zero")
        assert len(second["samples"]) == 2
        for i, sample in enumerate(second["samples"]):
            before = wrapped.calls[1][0][i]
            euler = before + .25 * before * (.5 - .75)
            after = wrapped.calls[2][0][i]
            want = ((after-euler).norm() / (euler-before).norm()).item()
            assert sample["scg_over_euler"] == pytest.approx(want, abs=1e-5)
            assert sample["bands"]["gain_applied"][0] == 1
            assert len(sample["bands"]["power_post"]) == len(header["bin_counts"])
        if header["debug_log"] == "jsonl+latents":
            with np.load(path.parent / "step_0003.npz", allow_pickle=False) as states:
                assert np.array_equal(states["euler"], expected.numpy())
        else:
            assert not list(path.parent.glob("*.npz"))


def test_interrupted_debug_run_keeps_completed_steps(nodes, inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    model, _, _, x = inputs
    def stop(event):
        if event["i"] == 1:
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        sampler(nodes, debug_log="jsonl")(KSamplerX0Inpaint(model), x, torch.tensor([1., .5, 0.]),
                                         callback=stop, disable=True)
    path, = (tmp_path / "moss_scg_debug").glob("*/trace.jsonl")
    records = [json.loads(s) for s in path.read_text().splitlines()]
    assert records[-1]["status"] == "error" and records[-1]["steps_written"] == 1


def test_paired_trace_analysis_follows_model_response_after_first_correction(nodes, inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    model, _, _, x = inputs
    sigmas = torch.tensor([1., .75, .5, 0.])
    for strength in (0, 1):
        loop = sampler(nodes, build(nodes, inputs), debug_log="jsonl+latents",
                       strength=strength, sigma_min=.5, sigma_max=.5)
        loop(KSamplerX0Inpaint(model), x, sigmas, disable=True)
    compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/compare_scg_traces.py"))["compare"]
    runs = {}
    for path in (tmp_path / "moss_scg_debug").glob("*/trace.jsonl"):
        runs[json.loads(path.read_text().splitlines()[0])["strength"]] = path
    first, second, third = list(compare(runs[0], runs[1]))
    assert first["post_ab_relative_l2"] == first["denoised_ab_relative_l2"] == 0
    assert second["denoised_ab_relative_l2"] == 0 and second["post_ab_relative_l2"] > 0
    assert third["denoised_ab_relative_l2"] > 0 and third["scg_over_euler"] == 0
    content = runs[0].read_text()
    runs[0].write_text(content.replace('"sigmas":[1.0,', '"sigmas":[0.9,'))
    with pytest.raises(ValueError, match="sigmas"):
        list(compare(runs[0], runs[1]))


def test_debug_preserves_clamp_and_per_sample_fallback_evidence(nodes, inputs, tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    model, _, _, x = inputs
    profile = build(nodes, inputs)
    profile = replace(profile, power=profile.power * 256)
    sigmas = torch.tensor([1., .5, 0.])
    loop = sampler(nodes, profile, strength=1, debug_log="jsonl")
    loop(KSamplerX0Inpaint(model), torch.cat((x, x * torch.nan, x * 0)), sigmas, disable=True)
    path, = (tmp_path / "moss_scg_debug").glob("*/trace.jsonl")
    def reject(value):
        pytest.fail(f"JSON contained nonstandard {value}")
    step = json.loads(path.read_text().splitlines()[1], parse_constant=reject)
    clipped, bad, zero = step["samples"]
    band = max(range(1, len(profile.power)), key=lambda k: profile.power[k].item())
    assert clipped["bands"]["gain_requested"][band] > 2
    assert clipped["bands"]["clamped"][band] and clipped["bands"]["gain_applied"][band] == 2
    assert bad["fallback_to_euler"] and bad["nonfinite_post"] == x.numel()
    assert all(g == 1 for g in bad["bands"]["gain_applied"])
    assert zero["scg_over_euler"] is None and not any(zero["bands"]["valid"])


def test_experiment_uses_recorded_budget_and_compares_nonzero_control(nodes, inputs, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    model, _, _, _ = inputs
    x = torch.randn(2,128,5,7, generator=torch.Generator().manual_seed(42))
    sigmas = torch.tensor([1., .8, .6, .4, 0.])
    def run(**options):
        loop = sampler(nodes, strength=.45, debug_log="jsonl+latents", **options)
        return loop(KSamplerX0Inpaint(model), x, sigmas, disable=True)
    original = run()
    root = tmp_path / "moss_scg_debug"
    control, = root.iterdir()
    altered = run(within_bin_mix=.5, budget_trace=control.name)
    assert not torch.equal(original, altered)
    candidate, = [d for d in root.iterdir() if d != control]
    compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/compare_scg_traces.py"))["compare"]
    rows = list(compare(control, candidate))
    assert len(rows) == 8 and all(row["batch_denoised_identical"] for row in rows[:2])
    for r in rows[:6]:
        assert r["scg_delta_l2"] == pytest.approx(r["control_scg_l2"], rel=1e-4)
    assert rows[-1]["post_ab_relative_l2"] > 0
    records = [json.loads(line) for line in (candidate / "trace.jsonl").read_text().splitlines()]
    assert records[0]["within_bin_mix"] == .5 and records[0]["budget_trace"] == control.name
    assert "radial_gain_control" in records[1]["samples"][0]["bands"]
    original_steps = [json.loads(line) for line in (control / "trace.jsonl").read_text().splitlines()][1:-1]
    for base, step in zip(original_steps, records[1:-1]):
        for a, b in zip(base["samples"], step["samples"]):
            actual = torch.tensor(b["bands"]["power_scg_delta"][1:])
            requested = torch.tensor(a["bands"]["power_scg_delta"][1:])
            torch.testing.assert_close(actual, requested, rtol=1e-4, atol=1e-10)
    with pytest.raises(ValueError, match="sigma_min"):
        run(within_bin_mix=.5, budget_trace=control.name, sigma_min=.3)
