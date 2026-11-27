"""ComfyUI V3 nodes for local MOSS Euler/LEAP calibration and measurement."""

from dataclasses import dataclass
import json
import logging
import math
from time import perf_counter
from weakref import ref

import torch
from comfy_api.latest import io

from .leap import LeapAccumulator, LeapBank, prediction_steps, sample_euler_reference, sample_leap_euler
from .nodes import _validate_model


_LOG = logging.getLogger(__name__)
_MAX_SEED = (1 << 64) - 1
_MAX_NONCE = (1 << 53) - 1
_DEFAULT_ID = "moss-image-sft-int8-convrot"
_PATCHER_OVERRIDES = (
    "patches", "weight_wrapper_patches", "wrappers", "callbacks",
    "injections", "additional_models", "hook_patches", "forced_hooks",
)
_UNSUPPORTED_CONDITIONING = (
    "area", "mask", "control", "hooks", "gligen", "reference_latents",
    "start_percent", "end_percent", "start_sigma", "end_sigma",
)


@dataclass(frozen=True)
class _BoundBank:
    model_ref: ref
    calibration_positive: torch.Tensor
    calibration_seeds: tuple[int, ...]
    bank: LeapBank


def _class_name(value) -> str:
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def _seed(value, *, label: str, max_value: int = _MAX_SEED) -> int:
    if type(value) is not int or not 0 <= value <= max_value:
        raise ValueError(f"MOSS LEAP {label} must be an integer from 0 to {max_value}.")
    return value


def _conditioning(values):
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError("MOSS LEAP requires one native text conditioning entry per side.")
    item = values[0]
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise ValueError("MOSS LEAP requires native text conditioning.")
    tensor, options = item
    if not torch.is_tensor(tensor) or not isinstance(options, dict):
        raise ValueError("MOSS LEAP requires native text conditioning.")
    if any(options.get(key) is not None for key in _UNSUPPORTED_CONDITIONING):
        raise ValueError("MOSS LEAP does not support regional, edited, controlled or scheduled conditioning.")
    return tensor


def _prepare(model, positive, negative, sigmas, latent_image, cfg, checkpoint_id):
    from comfy import sample
    from comfy.latent_formats import Flux2

    _validate_model(model, feature="MOSS LEAP")
    if (
        any(getattr(model, key, None) for key in _PATCHER_OVERRIDES)
        or any(key != "model_sampling" for key in getattr(model, "object_patches", {}))
        or any(model.model_options.get(key) for key in ("model_function_wrapper", "wrappers", "multigpu"))
        or any(model.model_options.get("transformer_options", {}).get(key)
               for key in ("patches", "patches_replace", "wrappers", "callbacks", "rope_options"))
    ):
        raise ValueError("MOSS LEAP requires an unpatched native model.")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("MOSS LEAP requires a non-empty declared checkpoint identifier.")
    if type(cfg) not in (float, int) or not math.isfinite(cfg) or not 0 <= cfg <= 100:
        raise ValueError("MOSS LEAP CFG must be finite and in [0, 100].")
    device = torch.device(model.load_device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("MOSS LEAP currently measures CPU or PyTorch CUDA/HIP sampling only.")
    p = _conditioning(positive)
    _conditioning(negative)

    if not isinstance(latent_image, dict) or "samples" not in latent_image:
        raise ValueError("MOSS LEAP requires a native empty LATENT input.")
    if latent_image.get("noise_mask") is not None:
        raise ValueError("MOSS LEAP does not support a noise mask.")
    if latent_image.get("batch_index") is not None:
        raise ValueError("MOSS LEAP supports one latent without batch_index.")
    latent_format = model.get_model_object("latent_format")
    latent = latent_image["samples"]
    if not isinstance(latent_format, Flux2) or not torch.is_tensor(latent) or latent.is_nested:
        raise ValueError("MOSS LEAP requires MOSS packed Flux2 latent format.")
    if (latent.ndim != 4 or tuple(latent.shape[:2]) != (1, 128)
            or not latent.is_floating_point() or any(size < 1 for size in latent.shape)):
        raise ValueError("MOSS LEAP requires an empty, packed Flux2 LATENT of shape [1, 128, H, W].")
    if torch.count_nonzero(latent).item():
        raise ValueError("MOSS LEAP requires an empty LATENT for full text-to-image denoising.")
    fixed = sample.fix_empty_latent_channels(
        model, latent, latent_image.get("downscale_ratio_spacial"),
        latent_image.get("downscale_ratio_temporal"),
    )
    if fixed.shape != latent.shape:
        raise ValueError("MOSS LEAP does not resize or reinterpret its source LATENT.")
    latent_data = {**latent_image, "samples": fixed}

    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("MOSS LEAP requires a one-dimensional floating sigma schedule.")
    grid = sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()
    sampling = model.get_model_object("model_sampling")
    if (len(grid) < 2 or not all(math.isfinite(s) for s in grid)
            or not all(a > b for a, b in zip(grid, grid[1:]))
            or grid[-1] != 0.0
            or not math.isclose(grid[0], float(sampling.sigma_max), rel_tol=1e-5)):
        raise ValueError("MOSS LEAP requires a full strictly descending sigma schedule ending at zero.")

    diffusion = model.get_model_object("diffusion_model")
    meta = {
        "checkpoint_id": checkpoint_id.strip(),
        "cfg": float(cfg),
        "sampler": "native-euler",
        "sampling_class": _class_name(sampling),
        "shift": float(sampling.shift),
        "multiplier": float(sampling.multiplier),
        "noise_scale": float(sampling.noise_scale),
        "latent_format": _class_name(latent_format),
        "latent_dtype": str(fixed.dtype),
        "sigma_dtype": str(sigmas.dtype),
        "model_class": _class_name(diffusion),
        "model_dtype": str(diffusion.dtype),
        "compute_dtype": str(model.model.get_dtype_inference()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
    }
    if not (math.isfinite(meta["shift"]) and meta["shift"] > 0
            and math.isfinite(meta["noise_scale"]) and meta["noise_scale"] > 0):
        raise ValueError("MOSS LEAP requires finite positive native flow parameters.")
    return latent_data, p, grid, device, diffusion, meta


def _sync(device):
    # ROCm builds use the torch.cuda namespace, too.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _Observation:
    def __init__(self):
        self.guided_calls = 0
        self.forward_batch_sizes = []
        self.forward_cfg_batches = []

    def wrap(self, sampler_fn):
        def sample_with_counts(native_guide, x, sigmas, **options):
            def measured_guide(*args, **kwargs):
                self.guided_calls += 1
                return native_guide(*args, **kwargs)
            return sampler_fn(measured_guide, x, sigmas, **options)
        sample_with_counts.__name__ = f"measured_{sampler_fn.__name__}"
        return sample_with_counts

    def record_forward(self, module, args, kwargs):
        x = args[0] if args else kwargs["x"]
        self.forward_batch_sizes.append(x.shape[0])
        labels = kwargs.get("transformer_options", {}).get("cond_or_uncond")
        self.forward_cfg_batches.append(list(labels) if labels is not None else None)

    def fields(self):
        return {
            "guided_calls": self.guided_calls,
            "physical_forwards": len(self.forward_batch_sizes),
            "forward_batch_sizes": self.forward_batch_sizes,
            "forward_cfg_batches": self.forward_cfg_batches,
        }


def _sample(model, positive, negative, cfg, latent, noise, seed, sigmas, sampler_fn,
            diffusion, device, *, measure=True):
    from comfy import model_management, samplers

    guider = samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    observation = _Observation()
    sampler = samplers.KSAMPLER(observation.wrap(sampler_fn))
    handle = diffusion.register_forward_pre_hook(observation.record_forward, with_kwargs=True)
    try:
        if measure:
            _sync(device)
            start = perf_counter()
        result = guider.sample(
            noise, latent, sampler, sigmas, denoise_mask=None,
            callback=lambda *args: model_management.throw_exception_if_processing_interrupted(),
            disable_pbar=True, seed=seed,
        )
        if measure:
            _sync(device)
            elapsed = perf_counter() - start
        else:
            elapsed = None
    finally:
        handle.remove()
    if observation.guided_calls < 1 or len(observation.forward_batch_sizes) < observation.guided_calls:
        raise RuntimeError("MOSS LEAP run did not observe the native denoiser; do not time this result.")
    return result, observation, elapsed


def _report(**fields):
    payload = json.dumps({"schema_version": 1, **fields}, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _LOG.info("MOSS guidance report %s", payload)
    return payload


class MossImageLEAPCalibrate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bank_io = io.Custom("MOSS_LEAP_BANK")
        return io.Schema(
            node_id="MossImageLEAPCalibrate",
            display_name="MOSS-Image LEAP Calibrate",
            category="model/sampling/moss_image",
            description="Fit LEAP locally from full Euler-25 MOSS trajectories. Bank is session-local.",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"), io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Float.Input("cfg", default=2.5, min=0.0, max=100.0, step=0.1),
                io.String.Input("checkpoint_id", default=_DEFAULT_ID),
                io.Int.Input("calibration_seed", default=1000, min=0, max=_MAX_SEED - 31),
                io.Int.Input("reference_count", default=2, min=2, max=32, step=1),
            ],
            outputs=[bank_io.Output(display_name="bank"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, positive, negative, sigmas, latent_image, cfg=2.5,
                checkpoint_id=_DEFAULT_ID, calibration_seed=1000, reference_count=2):
        from comfy import sample

        if type(reference_count) is not int or not 2 <= reference_count <= 32:
            raise ValueError("MOSS LEAP reference_count must be an integer from 2 to 32.")
        seed = _seed(calibration_seed, label="calibration_seed", max_value=_MAX_SEED - reference_count + 1)
        lat_data, prompt, grid, device, diffusion, meta = _prepare(
            model, positive, negative, sigmas, latent_image, cfg, checkpoint_id,
        )
        if len(grid) != 26:
            raise ValueError("MOSS LEAP calibration needs the 25-step Euler sigma grid.")
        latent = lat_data["samples"]
        accumulator = LeapAccumulator(sigmas, latent_shape=latent.shape, metadata=meta)
        _sync(device)
        started = perf_counter()
        observations = []
        seeds = tuple(range(seed, seed + reference_count))
        for reference_seed in seeds:
            velocities = []
            def capture(i, d):
                velocities.append(d.detach().to("cpu", copy=True))

            def reference_sampler(guide, x, active_sigmas, **kwargs):
                return sample_euler_reference(guide, x, active_sigmas, on_velocity=capture, **kwargs)

            _, observation, _ = _sample(
                model, positive, negative, float(cfg), latent,
                sample.prepare_noise(latent, reference_seed), reference_seed,
                sigmas, reference_sampler, diffusion, device, measure=False,
            )
            if observation.guided_calls != 25:
                raise RuntimeError("MOSS LEAP calibration did not execute the entire Euler reference.")
            accumulator.add_trajectory(velocities)
            observations.append(observation)
        bank = accumulator.finish()
        _sync(device)
        calibration_seconds = perf_counter() - started
        bound = _BoundBank(ref(model), prompt.detach().to("cpu").clone(), seeds, bank)
        report = _report(
            mode="calibrate", calibration_seconds=calibration_seconds,
            reference_count=reference_count, calibration_seeds=list(seeds),
            guided_calls=sum(o.guided_calls for o in observations),
            physical_forwards=sum(len(o.forward_batch_sizes) for o in observations),
            reference_runs=[o.fields() for o in observations], steps=len(grid) - 1,
            sigmas=grid, latent_shape=list(latent.shape), **meta,
        )
        return io.NodeOutput(bound, report, ui={"text": [report]})


class MossImageLEAP(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bank_io = io.Custom("MOSS_LEAP_BANK")
        return io.Schema(
            node_id="MossImageLEAP",
            display_name="MOSS-Image LEAP / Euler (Benchmark)",
            category="model/sampling/moss_image",
            description="Compare native Euler and LEAP with actual denoiser calls and synchronized sampler time.",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"), io.Noise.Input("noise"),
                io.Sigmas.Input("sigmas"), io.Latent.Input("latent_image"),
                io.Combo.Input("mode", options=["euler", "leap"], default="euler"),
                io.Float.Input("cfg", default=2.5, min=0.0, max=100.0, step=0.1),
                io.String.Input("checkpoint_id", default=_DEFAULT_ID),
                io.Int.Input("run_nonce", default=0, min=0, max=_MAX_NONCE,
                             tooltip="Change for a new measured execution without changing the seed or noise."),
                bank_io.Input("bank", optional=True),
            ],
            outputs=[io.Latent.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, positive, negative, noise, sigmas, latent_image, mode="euler",
                cfg=2.5, checkpoint_id=_DEFAULT_ID, run_nonce=0, bank=None):
        from comfy import model_management, samplers
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        if mode not in ("euler", "leap"):
            raise ValueError("MOSS LEAP mode must be euler or leap.")
        if mode == "leap" and not isinstance(bank, _BoundBank):
            raise ValueError("MOSS LEAP mode needs a bank from MOSS-Image LEAP Calibrate.")
        if not isinstance(noise, Noise_RandomNoise):
            raise ValueError("MOSS LEAP requires ComfyUI's native RandomNoise node.")
        nonce = _seed(run_nonce, label="run_nonce", max_value=_MAX_NONCE)
        seed = _seed(noise.seed, label="RandomNoise seed")
        lat_data, prompt, grid, device, diffusion, meta = _prepare(
            model, positive, negative, sigmas, latent_image, cfg, checkpoint_id,
        )
        latent = lat_data["samples"]
        steps = len(grid) - 1
        if mode == "leap":
            if bank.model_ref() is not model:
                raise ValueError("MOSS LEAP requires the same loaded MODEL as the calibration bank.")
            if seed in bank.calibration_seeds:
                raise ValueError("MOSS LEAP evaluation seed must differ from the calibration seeds.")
            if torch.equal(prompt.detach().to("cpu"), bank.calibration_positive):
                raise ValueError("MOSS LEAP evaluation positive prompt must differ from the calibration prompt.")
            bank.bank.validate(sigmas, latent.shape, meta)
            leap_steps = prediction_steps(steps, bank.bank.window)
            def sampler_fn(guide, x, active_sigmas, **kwargs):
                return sample_leap_euler(guide, x, active_sigmas, bank=bank.bank, **kwargs)
        else:
            leap_steps = ()
            sampler_fn = samplers.sampler_object("euler").sampler_function
        sampled, observed, seconds = _sample(
            model, positive, negative, float(cfg), latent, noise.generate_noise(lat_data),
            seed, sigmas, sampler_fn, diffusion, device,
        )
        if observed.guided_calls != steps - len(leap_steps):
            raise RuntimeError("MOSS LEAP observed guided evaluations do not match the scheduled calls.")
        if not torch.isfinite(sampled).all().item():
            raise RuntimeError("MOSS LEAP produced a non-finite latent; reject this measurement.")
        out = {**lat_data, "samples": sampled.to(model_management.intermediate_device())}
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        report = _report(
            mode=mode, run_nonce=nonce, seed=seed, steps=steps, sigmas=grid,
            predicted_steps=list(leap_steps), latent_shape=list(latent.shape),
            sampler_seconds=seconds, timer_boundary="native_cfg_guider_sample",
            calibration_seeds=list(bank.calibration_seeds) if mode == "leap" else None,
            **observed.fields(), **meta,
        )
        return io.NodeOutput(out, report, ui={"text": [report]})
