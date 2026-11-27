"""Exercise node entry points against a small, independent Comfy boundary."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native(monkeypatch):
    events = []
    cpu = torch.device("cpu")
    grid25 = torch.linspace(1.0, 0.0, 26)
    latent = {"samples": torch.zeros(1, 128, 2, 2), "downscale_ratio_spacial": 16}
    positive = [[torch.ones(1, 2, 3), {}]]
    calibration_positive = [[torch.full((1, 2, 3), 1.5), {}]]
    negative = [[torch.zeros(1, 2, 3), {}]]

    def register(path, **atts):
        value = ModuleType(path)
        value.__dict__.update(atts)
        monkeypatch.setitem(sys.modules, path, value)
        return value

    class NativeMoss(torch.nn.Module):
        dtype = torch.float32

        def forward(self, x, sigma, context, *, transformer_options):
            events.append("forward")
            return x - sigma[:, None, None, None] * context[:, 0, 0][:, None, None, None]

    class DiscreteFlow:
        multiplier = 1.0
        shift = 5.0

    class ConstantPrediction:
        def calculate_denoised(self, sigma, model_output, model_input):
            return model_input - model_output * sigma

    class NativeFlow(DiscreteFlow, ConstantPrediction):
        sigma_max = 1.0
        noise_scale = 1.0

    class Flux2:
        latent_channels = 128
        latent_dimensions = 2

    class ComfyPatcher:
        def __init__(self):
            self.model = SimpleNamespace(
                diffusion_model=NativeMoss(), model_sampling=NativeFlow(),
                get_dtype_inference=lambda: torch.float32, latent_format=Flux2(),
            )
            self.model_options = {}
            self.object_patches = {}
            self.load_device = cpu
            self.patches = {}

        def get_model_object(self, name):
            return self.object_patches.get(name, getattr(self.model, name))

    class Guider:
        def __init__(self, model):
            self.model_patcher = model

        def set_conds(self, positive, negative):
            self.positive = positive[0][0]
            self.negative = negative[0][0]

        def set_cfg(self, cfg):
            self.cfg = cfg

        def sample(self, noise, latent, sampler, sigmas, **kw):
            events.append("guider")
            inner = self.model_patcher.get_model_object("diffusion_model")
            def model(x, ss, **unused):
                events.append("guided")
                if self.cfg == 1:
                    return inner(x, ss, self.positive, transformer_options={"cond_or_uncond": [0]})
                if self.positive.shape == self.negative.shape:
                    a, b = inner(torch.cat([x, x]), ss.repeat(2),
                                 torch.cat([self.positive, self.negative]),
                                 transformer_options={"cond_or_uncond": [0, 1]}).chunk(2)
                else:
                    a = inner(x, ss, self.positive, transformer_options={"cond_or_uncond": [0]})
                    b = inner(x, ss, self.negative, transformer_options={"cond_or_uncond": [1]})
                return b + (a - b) * self.cfg

            return sampler.sampler_function(model, noise, sigmas,
                                            callback=kw.get("callback"),
                                            disable=kw.get("disable_pbar"))

    def native_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
        for i, (s, s_next) in enumerate(zip(sigmas, sigmas[1:])):
            denoised = model(x, s.expand(x.shape[0]))
            if callback is not None:
                callback({"x": x, "i": i, "sigma": s, "sigma_hat": s, "denoised": denoised})
            x = x + (x - denoised) * ((s_next - s) / s)
        return x

    def prepare_noise(samples, seed, noise_inds=None):
        events.append(("seed", seed))
        return torch.randn(samples.shape, generator=torch.Generator().manual_seed(seed))

    class Noise:
        def __init__(self, seed):
            self.seed = seed

        def generate_noise(self, lat):
            return prepare_noise(lat["samples"], self.seed)

    class KSAMPLER:
        def __init__(self, fn):
            self.sampler_function = fn

    sample = register("comfy.sample", prepare_noise=prepare_noise,
                      fix_empty_latent_channels=lambda model, x, *args: x)
    samplers = register("comfy.samplers", CFGGuider=Guider, KSAMPLER=KSAMPLER,
                        sampler_object=lambda name: KSAMPLER(native_euler))
    model_management = register("comfy.model_management", intermediate_device=lambda: cpu,
                                throw_exception_if_processing_interrupted=lambda: None)
    register("comfy", __path__=[], sample=sample, samplers=samplers,
             model_management=model_management)
    for mod in ("comfy.ldm", "comfy.ldm.moss_image", "comfy_api", "comfy_extras"):
        register(mod, __path__=[])
    register("comfy.ldm.moss_image.model", MOSSImage=NativeMoss)
    register("comfy.model_patcher", ModelPatcher=ComfyPatcher)
    register("comfy.model_sampling", ModelSamplingDiscreteFlow=DiscreteFlow, CONST=ConstantPrediction)
    register("comfy.latent_formats", Flux2=Flux2)
    register("comfy_extras.nodes_custom_sampler", Noise_RandomNoise=Noise)
    def port(name=None, **kw):
        return SimpleNamespace(name=name, **kw)
    p = SimpleNamespace(Input=port, Output=port)
    io = SimpleNamespace(ComfyNode=type("ComfyNode", (), {}),
                         Schema=lambda **kw: SimpleNamespace(**kw),
                         NodeOutput=lambda *args, **kw: SimpleNamespace(result=args, **kw),
                         Custom=lambda name: p, **{
                             k:p for k in ("Model", "Conditioning", "Noise", "Sigmas", "Latent",
                                           "Float", "Int", "String", "Combo", "Boolean")
                         })
    register("comfy_api.latest", io=io, ComfyExtension=type("ComfyExtension", (), {}))

    name = "_moss_leap_adapter_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py",
                                                 submodule_search_locations=[str(ROOT)])
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, plugin)
    spec.loader.exec_module(plugin)
    yield SimpleNamespace(events=events, model=ComfyPatcher(), make_model=ComfyPatcher,
                          grid=grid25, latent=latent, Noise=Noise, positive=positive,
                          negative=negative, calibration_positive=calibration_positive,
                          module=f"{name}.leap_nodes")
    for k in tuple(sys.modules):
        if k.startswith(name + "."):
            sys.modules.pop(k, None)


def adapter(native):
    spec = importlib.util.find_spec(native.module)
    assert spec is not None, "LEAP adapter is missing"
    return importlib.import_module(native.module)


def run_calibration(n, **kw):
    inputs = dict(model=n.model, positive=n.calibration_positive, negative=n.negative,
                  sigmas=n.grid, latent_image=n.latent, cfg=2.5, checkpoint_id="sft-int8",
                  calibration_seed=11, reference_count=2)
    inputs.update(kw)
    return adapter(n).MossImageLEAPCalibrate.execute(**inputs)


def run(n, **kw):
    inputs = dict(model=n.model, positive=n.positive, negative=n.negative, sigmas=n.grid,
                  latent_image=n.latent, cfg=2.5, checkpoint_id="sft-int8",
                  noise=n.Noise(101), mode="euler", run_nonce=0)
    inputs.update(kw)
    result = adapter(n).MossImageLEAP.execute(**inputs)
    latent, report = result.result
    assert result.ui["text"] == [report]
    return latent["samples"], json.loads(report)


def test_calibration_runs_two_complete_reference_trajectories(native):
    bank, report = run_calibration(native).result
    assert json.loads(report)["guided_calls"] == 50
    assert native.events.count("guided") == 50
    assert native.events.count("forward") == 50
    assert [e for e in native.events if isinstance(e, tuple)] == [("seed", 11), ("seed", 12)]
    assert native.model.get_model_object("diffusion_model")._forward_pre_hooks == {}
    assert bank is not None


def test_three_arms_measure_actual_guided_and_physical_forward_counts(native):
    bank, _ = run_calibration(native).result
    base25, euler25 = run(native)
    _, euler15 = run(native, sigmas=torch.linspace(1., 0., 16))
    predicted, leap = run(native, mode="leap", bank=bank)
    assert [r["guided_calls"] for r in (euler25, euler15, leap)] == [25, 15, 15]
    assert [r["physical_forwards"] for r in (euler25, euler15, leap)] == [25, 15, 15]
    assert leap["forward_batch_sizes"] == [2] * 15
    assert all(sorted(v) == [0, 1] for v in leap["forward_cfg_batches"])
    assert leap["steps"] == 25 and leap["sampler_seconds"] >= 0.0 and leap["sigmas"] == euler25["sigmas"]
    torch.testing.assert_close(predicted, base25, atol=2e-5, rtol=1e-5)


def test_cfg1_and_longer_positive_require_different_physical_counts(native):
    shorter_neg = [[torch.zeros(1, 1, 3), {}]]
    _, cfg1 = run(native, cfg=1, negative=shorter_neg)
    _, split = run(native, negative=shorter_neg)
    assert cfg1["guided_calls"] == split["guided_calls"] == 25
    assert cfg1["physical_forwards"] == 25 and split["physical_forwards"] == 50
    assert cfg1["forward_cfg_batches"] == [[0]] * 25
    assert split["forward_cfg_batches"] == [[0], [1]] * 25


def test_nonce_does_not_change_noise_or_result(native):
    one, one_record = run(native, run_nonce=131)
    two, two_record = run(native, run_nonce=132)
    assert torch.equal(one, two)
    assert one_record["seed"] == two_record["seed"] == 101
    assert (one_record["run_nonce"], two_record["run_nonce"]) == (131, 132)
    assert native.events.count(("seed", 101)) == 2


@pytest.mark.parametrize("changed", ["model", "cfg", "sigmas", "checkpoint_id", "seed", "prompt"])
def test_calibration_contract_failure_does_not_start_sampling(native, changed):
    bank, _ = run_calibration(native).result
    alternatives = dict(model=native.make_model(), cfg=2.0, sigmas=torch.linspace(1., 0., 26)**2,
                        checkpoint_id="different-weights", noise=native.Noise(12),
                        positive=[[native.calibration_positive[0][0].clone(), {}]])
    kwargs = {("noise" if changed == "seed" else "positive" if changed == "prompt" else changed):
              alternatives["noise" if changed == "seed" else "positive" if changed == "prompt" else changed]}
    before = native.events.count("guider")
    with pytest.raises(ValueError):
        run(native, mode="leap", bank=bank, **kwargs)
    assert native.events.count("guider") == before


def test_unsupported_input_fails_before_guided_evaluation(native):
    with pytest.raises(ValueError, match="bank"):
        run(native, mode="leap")
    with pytest.raises(ValueError, match="mask"):
        run(native, latent_image={**native.latent, "noise_mask": torch.zeros(1)})
    with pytest.raises(ValueError, match="empty"):
        run(native, latent_image={"samples": torch.ones(1, 128, 2, 2)})
    with pytest.raises(ValueError, match="sigma"):
        run(native, sigmas=native.grid[1:])
    with pytest.raises(ValueError, match="conditioning"):
        run(native, positive=[[torch.ones(1, 2, 3), {"reference_latents": [torch.zeros(1)]}]])
    native.model.patches = {"block.weight":[1]}
    with pytest.raises(ValueError, match="patch"):
        run(native)
    assert "guided" not in native.events


def test_failed_native_forward_always_removes_physical_counter(native, monkeypatch):
    denoiser = native.model.get_model_object("diffusion_model")
    def fail(*args, **kwargs):
        raise RuntimeError("failed synthetic evaluation")
    monkeypatch.setattr(denoiser, "forward", fail)
    with pytest.raises(RuntimeError, match="failed synthetic evaluation"):
        run(native)
    assert not denoiser._forward_pre_hooks


def test_nonfinite_native_result_is_never_returned(native, monkeypatch):
    diffusion = native.model.get_model_object("diffusion_model")
    monkeypatch.setattr(diffusion, "forward", lambda x, *a, **kw: torch.full_like(x, float("nan")))
    with pytest.raises(RuntimeError, match="non-finite"):
        run(native)
    assert not diffusion._forward_pre_hooks


def test_cuda_hip_sync_wraps_native_sampler_and_validation_is_after_timer(native, monkeypatch):
    n = native
    n.model.load_device = torch.device("cuda:1")
    m = adapter(n)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "synthetic HIP GPU")
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda device: n.events.append(("sync", str(device))))
    def timer():
        n.events.append("timer")
        return float(len(n.events))
    monkeypatch.setattr(m, "perf_counter", timer)
    finite = torch.isfinite
    def check(t):
        n.events.append("finite")
        return finite(t)
    monkeypatch.setattr(torch, "isfinite", check)
    _, report = run(n)
    boundaries = [e for e in n.events if e in (("sync", "cuda:1"), "timer", "guider", "finite")]
    assert boundaries == [("sync", "cuda:1"), "timer", "guider", ("sync", "cuda:1"), "timer", "finite"]
    assert report["device"] == "cuda:1"


@pytest.mark.parametrize("flag", ["wrappers", "callbacks", "weight_wrapper_patches", "injections",
                                 "additional_models", "hook_patches"])
def test_native_patcher_side_overrides_fail_before_guided(native, flag):
    setattr(native.model, flag, {"nonstock": [object()]})
    with pytest.raises(ValueError, match="unpatched native model"):
        run(native)
    assert "guider" not in native.events
