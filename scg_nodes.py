"""ComfyUI V3 calibration and Euler adapter for native MOSS-Image SCG."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import logging
from pathlib import Path
import weakref

import torch
from tqdm.auto import trange

from comfy_api.latest import io

from .scg import (FFT_CONVENTION, SPECTRUM_SPACES, analytic_power, clean_power, correct,
                  from_spectral_space, radial_geometry, to_spectral_space)
from .scg_debug import DEBUG_MODES, _fp32_hash, load_budgets, open_trace, text_condition_hashes


_PROFILE_TYPE = "MOSS_SCG_PROFILE"
_UNCONNECTED = object()
_PACKED_CHANNELS = 128
_SPATIAL_SCALE = 16
_CONFLICTS = (
    "sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function",
    "sampler_calc_cond_batch_function", "model_function_wrapper", "denoise_mask_function", "wrappers",
)
_PATCHER_EXTENSIONS = ("wrappers", "injections", "additional_models", "callbacks", "hook_patches")


@dataclass(frozen=True, eq=False)
class SCGProfile:
    """In-memory only; tied to the identical native MODEL used by the guider."""

    power: torch.Tensor
    latent_shape: tuple[int, int, int]
    image_shape: tuple[int, int]
    reference_count: int
    vae_identity: int
    model_ref: weakref.ReferenceType
    model_identity: int
    diffusion_identity: int
    sampling_settings: tuple[float, float, float]
    fft_convention: str = FFT_CONVENTION
    spatial_power: torch.Tensor | None = None


def _guard_options(options):
    for name in _CONFLICTS:
        if options.get(name):
            raise ValueError(f"SCG cannot be combined with an existing {name} hook or wrapper.")
    transformer = options.get("transformer_options", {})
    for name in ("patches", "patches_replace", "wrappers"):
        if transformer.get(name):
            raise ValueError(f"SCG cannot be combined with transformer {name}.")


def _model_parts(patcher):
    from comfy.ldm.moss_image.model import MOSSImage
    from comfy.model_patcher import ModelPatcher
    from comfy.model_sampling import CONST, ModelSamplingDiscreteFlow

    if not isinstance(patcher, ModelPatcher):
        raise ValueError("SCG requires a native ComfyUI ModelPatcher.")
    diffusion = patcher.get_model_object("diffusion_model")
    if not isinstance(diffusion, MOSSImage) or hasattr(diffusion, "_orig_mod"):
        raise ValueError("SCG requires uncompiled native MOSS-Image.")
    sampling = patcher.get_model_object("model_sampling")
    if (
        not isinstance(sampling, ModelSamplingDiscreteFlow)
        or not isinstance(sampling, CONST)
        or getattr(sampling.calculate_denoised, "__func__", None) is not CONST.calculate_denoised
        or getattr(sampling.noise_scaling, "__func__", None) is not CONST.noise_scaling
    ):
        raise ValueError("SCG requires native discrete CONST flow prediction and noise scaling.")
    if sampling.multiplier != 1.0:
        raise ValueError("SCG requires a native MOSS-Image timestep multiplier of 1.")
    # LoRA and static BlockTune change denoiser weights. ComfyUI applies both
    # inside model(), before SCG corrects the subsequent Euler latent.
    for name in ("patches", "weight_wrapper_patches"):
        if any(not key.startswith("diffusion_model.") for key in getattr(patcher, name, {})):
            raise ValueError(f"SCG requires {name} to target diffusion_model parameters.")
    for name in _PATCHER_EXTENSIONS:
        if getattr(patcher, name, None):
            raise ValueError(f"SCG cannot be combined with ModelPatcher {name}.")
    if set(getattr(patcher, "object_patches", ())) - {"model_sampling"}:
        raise ValueError("SCG does not support object patches other than native-compatible model_sampling.")
    _guard_options(patcher.model_options)
    return diffusion, sampling


def _sampling_settings(sampling):
    shift, sigma_max, noise_scale = (
        float(sampling.shift), float(sampling.sigma_max), float(getattr(sampling, "noise_scale", 1.0))
    )
    if not all(math.isfinite(v) for v in (shift, sigma_max, noise_scale)) or not (
        shift > 0 and 0 < sigma_max <= 1 and noise_scale >= 0
    ):
        raise ValueError("SCG requires finite, valid MOSS flow sampling settings.")
    return shift, sigma_max, noise_scale


def _make_profile(model, vae, images) -> SCGProfile:
    from comfy.sd import VAE

    diffusion, sampling = _model_parts(model)
    if not isinstance(vae, VAE) or vae.latent_channels != _PACKED_CHANNELS or vae.downscale_ratio != _SPATIAL_SCALE:
        raise ValueError("SCG requires a ComfyUI 128-channel, 16× MOSS-compatible VAE.")
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or min(images.shape) == 0
            or images.shape[-1] != 3 or not images.is_floating_point()):
        raise ValueError("SCG requires a nonempty RGB image batch in ComfyUI BHWC format.")
    height, width = images.shape[1:3]
    if height % _SPATIAL_SCALE or width % _SPATIAL_SCALE:
        raise ValueError("SCG reference height and width must be divisible by 16; prepare the crop first.")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("SCG reference images must be finite.")
    latent = vae.encode(images)
    shape = (_PACKED_CHANNELS, height // _SPATIAL_SCALE, width // _SPATIAL_SCALE)
    if not isinstance(latent, torch.Tensor) or latent.shape != (images.shape[0], *shape):
        raise ValueError("SCG VAE output shape differs from the expected packed MOSS latent.")
    latent = model.model.process_latent_in(latent)
    if not isinstance(latent, torch.Tensor) or latent.shape != (images.shape[0], *shape):
        raise ValueError("SCG normalized reference does not preserve the packed latent shape.")
    geometry = radial_geometry(*shape[1:], latent.device)
    power = clean_power(latent, geometry).detach().to(device="cpu", copy=True)
    spatial = to_spectral_space(latent, "spatial_measured")
    spatial_power = clean_power(spatial, radial_geometry(*spatial.shape[-2:], spatial.device)).detach().cpu()
    return SCGProfile(
        power, shape, (height, width), images.shape[0], id(vae), weakref.ref(model), id(model.model),
        id(diffusion), _sampling_settings(sampling), spatial_power=spatial_power,
    )


@lru_cache(maxsize=1)
def _bundled_pair():
    record = json.loads((Path(__file__).parent / "calibration" / "flux2_1024.json").read_text(encoding="utf-8"))
    if record["fft_convention"] != FFT_CONVENTION:
        raise ValueError("Incompatible FFT convention in bundled SCG calibration.")
    return (tuple(record["packed_shape"]), len(record["photos"]),
            torch.tensor(record["packed_measured"], dtype=torch.float32),
            torch.tensor(record["spatial_measured"], dtype=torch.float32))


def _reference_for(space, profile, x, geometry):
    if space == "packed":
        ref = analytic_power(geometry) if profile is None else profile.power.to(x.device)
        return ref, None, None if profile is None else profile.reference_count
    if profile is None:
        shape, count, packed, spatial = _bundled_pair()
        if tuple(x.shape[1:]) != shape:
            raise ValueError("Bundled measured SCG spectra are for 1024x1024; connect a matching SCG Profile.")
    else:
        count, packed, spatial = profile.reference_count, profile.power, profile.spatial_power
        if spatial is None:
            raise ValueError("Rebuild the SCG Profile to measure both coordinate systems.")
    pair_id = _fp32_hash(torch.cat((packed, spatial)).float().cpu())
    reference = spatial if space == "spatial_measured" else packed
    return reference.to(x.device), pair_id, count


def _runtime_sampler(model, latent, sigmas, extra_args, profile):
    from comfy.samplers import CFGGuider, KSamplerX0Inpaint

    if not isinstance(model, KSamplerX0Inpaint) or not isinstance(model.inner_model, CFGGuider):
        raise ValueError("SCG requires a native ComfyUI guided sampler.")
    guider = model.inner_model
    if getattr(guider.predict_noise, "__func__", None) is not CFGGuider.predict_noise:
        raise ValueError("SCG requires ComfyUI's standard CFG or basic guider.")
    patcher = guider.model_patcher
    if profile is not None and (profile.model_ref() is not patcher or profile.model_identity != id(guider.inner_model)):
        raise ValueError("SCG profile must use the same MODEL instance as the sampler's guider.")
    diffusion, sampling = _model_parts(patcher)
    if guider.inner_model.diffusion_model is not diffusion or guider.inner_model.model_sampling is not sampling:
        raise ValueError("SCG requires the guider's effective MOSS model and sampling objects.")
    settings = _sampling_settings(sampling)
    if latent.ndim != 4 or min(latent.shape) == 0 or latent.shape[1] != _PACKED_CHANNELS or not latent.is_floating_point():
        raise ValueError("SCG requires a nonempty floating point NCHW latent with 128 channels.")
    if profile is not None:
        if id(diffusion) != profile.diffusion_identity or settings != profile.sampling_settings:
            raise ValueError("SCG profile model or sampling settings changed; rebuild the reference profile.")
        if profile.fft_convention != FFT_CONVENTION or tuple(latent.shape[1:]) != profile.latent_shape:
            raise ValueError("SCG profile does not match the sample latent shape or FFT convention.")
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("SCG does not support a denoise mask.")
    _guard_options(extra_args.get("model_options", {}))
    if model.latent_image is not None and bool(torch.count_nonzero(model.latent_image)):
        raise ValueError("SCG supports empty-latent T2I only, not image-to-image sampling.")
    for group in guider.conds.values():
        for condition in group or ():
            model_conds = condition.get("model_conds", {})
            if (
                condition.get("reference_latents") is not None or condition.get("concat_latent_image") is not None
                or any(key.startswith("ref_latents") or key in {"ref_kv", "c_concat"} for key in model_conds)
            ):
                raise ValueError("SCG does not support conditioned reference latents or image edit.")
            if condition.get("hooks") is not None or condition.get("control") is not None:
                raise ValueError("SCG does not support conditioning hooks or control models.")
    values = sigmas.detach().cpu().tolist() if sigmas.ndim == 1 else []
    if (
        len(values) < 2 or values[-1] != 0.0 or not math.isclose(values[0], settings[1], abs_tol=1e-7)
        or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        or not all(a > b for a, b in zip(values, values[1:]))
    ):
        raise ValueError("SCG requires the full, descending positive sigma grid ending at zero (denoise=1).")
    return values, settings[2]


def _trace_metadata(model, profile, args, values, strength, sigma_min, sigma_max, debug_log, within_bin_mix, budget_trace):
    guider = model.inner_model
    patcher = guider.model_patcher
    sampling = guider.inner_model.model_sampling
    loader = getattr(patcher, "cached_patcher_init", None)
    model_file = loader[1][0] if loader and loader[1] else None
    try:
        from comfy_execution.utils import get_executing_context
    except ImportError:
        context = None
    else:
        context = get_executing_context()
    return dict(
        within_bin_mix=within_bin_mix, budget_trace=budget_trace,
        debug_log=debug_log, seed=args.get("seed"), cfg=getattr(guider, "cfg", None),
        strength=strength, sigma_min=sigma_min, sigma_max=sigma_max, sigmas=values,
        noise_scale=float(getattr(sampling, "noise_scale", 1.0)), shift=float(sampling.shift),
        reference_kind="analytic" if profile is None else "measured",
        reference_count=None if profile is None else profile.reference_count,
        model=type(guider.inner_model.diffusion_model).__name__,
        model_file=Path(model_file).name if isinstance(model_file, (str, Path)) else None,
        patches_uuid=str(patcher.patches_uuid) if hasattr(patcher, "patches_uuid") else None,
        patched_weights=len(getattr(patcher, "patches", {})),
        weight_wrappers=len(getattr(patcher, "weight_wrapper_patches", {})),
        block_tune=patcher.model_options.get("transformer_options", {}).get("moss_block_tune_preset"),
        text_conditions=text_condition_hashes(guider.conds),
        execution=context._asdict() if context is not None else None,
    )


def _make_sampler(profile, strength, sigma_min, sigma_max, debug_log="off", within_bin_mix=0., budget_trace="",
                  spectrum_space="packed"):
    from comfy.k_diffusion.sampling import to_d

    @torch.no_grad()
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        args = {} if extra_args is None else extra_args
        values, noise_scale = _runtime_sampler(model, x, sigmas, args, profile)
        active = [0 < sigma and sigma_min <= sigma <= sigma_max for sigma in values[1:]]
        tracing = debug_log != "off"
        context = nullcontext()
        budgets = None
        if any(active) or tracing or budget_trace:
            height, width = x.shape[-2:]
            factor = 2 if spectrum_space == "spatial_measured" else 1
            geometry = radial_geometry(height * factor, width * factor, x.device)
            reference, pair_id, count = _reference_for(spectrum_space, profile, x, geometry)
        total_only = bool(budget_trace and spectrum_space == "spatial_measured" and not within_bin_mix)
        if tracing or budget_trace:
            metadata = _trace_metadata(model, profile, args, values, strength, sigma_min, sigma_max,
                                       debug_log, within_bin_mix, budget_trace)
            metadata.update(spectrum_space=spectrum_space, paired_reference_id=pair_id, reference_count=count,
                            budget_mode="total_l2" if total_only else ("per_ring" if budget_trace else "local"))
            if pair_id:
                metadata["reference_kind"] = "bundled_measured" if profile is None else "measured"
            if budget_trace:
                budgets, first_hash = load_budgets(budget_trace, metadata, reference, x)
                metadata["budget_first_denoised_sha256_fp32"] = first_hash
            if tracing:
                context = open_trace(metadata, reference, x)
        s_in = x.new_ones([x.shape[0]])
        with context as trace:
            for i in trange(len(values) - 1, disable=disable):
                sigma = sigmas[i]
                denoised = model(x, sigma * s_in, **args)
                if budgets is not None and i == 0 and _fp32_hash(denoised.detach().float().cpu()) != first_hash:
                    logging.warning("[MOSS SCG] First model prediction differs from the budget control before SCG.")
                d = to_d(x, sigma, denoised)
                if callback is not None:
                    callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
                before = x if tracing else None
                x = x + d * (sigmas[i + 1] - sigma)
                euler = x if tracing else None
                audit = None
                if active[i] and strength:
                    audit = {} if tracing else None
                    budget = None if budgets is None else budgets[i]
                    corrected = correct(to_spectral_space(x, spectrum_space), reference, geometry, values[i + 1],
                                        noise_scale, strength, audit=audit, within_bin_mix=within_bin_mix,
                                        budget_power=None if total_only else budget, total_l2=budget if total_only else None)
                    x = from_spectral_space(corrected, spectrum_space)
                    del corrected
                if trace is not None:
                    if values[i + 1] == 0:
                        reason = "terminal"
                    elif not strength:
                        reason = "strength_zero"
                    elif active[i]:
                        reason = "applied"
                    else:
                        reason = "outside_window"
                    trace.step(i + 1, values[i], values[i + 1], reason, before, denoised, euler, x, audit)
                    del before, euler, audit
        return x

    return sample


class MossImageSCGProfile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImageSCGProfile", display_name="MOSS-Image SCG Profile",
            category="model/sampling/moss_image",
            description="Build an in-memory reference spectrum from prepared, same-size RGB images.",
            inputs=[io.Model.Input("model"), io.Vae.Input("vae"), io.Image.Input("images")],
            outputs=[io.Custom(_PROFILE_TYPE).Output("profile")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, model, vae, images):
        return io.NodeOutput(_make_profile(model, vae, images))


class MossImageSCGEuler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImageSCGEuler", display_name="MOSS-Image SCG Euler",
            category="model/sampling/moss_image",
            description="Correct the post-Euler latent in packed or unpacked spatial coordinates. Use the built-in reference or connect an image-calibrated SCG Profile.",
            inputs=[
                io.Custom(_PROFILE_TYPE).Input("profile", optional=True, lazy=True),
                io.Float.Input("strength", default=.15, min=0.0, max=1.0, step=.01),
                io.Float.Input("sigma_min", default=.10, min=0.0, max=1.0, step=.01),
                io.Float.Input("sigma_max", default=.95, min=0.0, max=1.0, step=.01),
                io.Combo.Input("debug_log", options=list(DEBUG_MODES), default="off", optional=True,
                               tooltip="Write per-step diagnostics to output/moss_scg_debug. +latents also saves full FP32 states."),
                io.Float.Input("within_bin_mix", default=0., min=0., max=1., step=.05, optional=True,
                               tooltip="0 uses radial SCG. Higher values move amplitudes toward a common value within each ring at matched correction energy."),
                io.String.Input("budget_trace", default="", optional=True,
                                tooltip="Radial control run in output/moss_scg_debug. Spatial radial mode matches only its total L2 per sample and step."),
                io.Combo.Input("spectrum_space", options=list(SPECTRUM_SPACES), default="packed", optional=True,
                               tooltip="packed keeps the original prior. The two measured modes use paired spectra from the same photos; spatial undoes FLUX.2 2x2 packing for the FFT."),
            ],
            outputs=[io.Sampler.Output("sampler")],
        )

    @classmethod
    def check_lazy_status(cls, profile=_UNCONNECTED, strength=.15, sigma_min=.10, sigma_max=.95, debug_log="off",
                          within_bin_mix=0., budget_trace="", spectrum_space="packed"):
        # ComfyUI omits unconnected optional inputs and passes None for a pending link.
        return ["profile"] if (strength != 0 or debug_log != "off") and profile is None else []

    @classmethod
    def execute(cls, profile=None, strength=.15, sigma_min=.10, sigma_max=.95, debug_log="off",
                within_bin_mix=0., budget_trace="", spectrum_space="packed"):
        from comfy.samplers import KSAMPLER, ksampler

        strength = float(strength)
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("SCG strength must be finite and within [0, 1].")
        if debug_log not in DEBUG_MODES:
            raise ValueError(f"Unknown SCG debug_log mode: {debug_log!r}")
        within_bin_mix = float(within_bin_mix)
        budget_trace = budget_trace.strip()
        if not 0 <= within_bin_mix <= 1:
            raise ValueError("SCG within_bin_mix must be finite and within [0, 1].")
        if spectrum_space not in SPECTRUM_SPACES:
            raise ValueError(f"Unknown SCG spectrum_space: {spectrum_space!r}")
        if budget_trace and not (strength and (within_bin_mix or spectrum_space == "spatial_measured")):
            raise ValueError("SCG budget_trace requires nonzero strength and either spatial_measured or within_bin_mix.")
        if strength == 0 and debug_log == "off":
            return io.NodeOutput(ksampler("euler"))
        if profile is not None and not isinstance(profile, SCGProfile):
            raise ValueError("SCG expected a MOSS reference profile.")
        sigma_min, sigma_max = float(sigma_min), float(sigma_max)
        if not (0 <= sigma_min <= sigma_max <= 1):
            raise ValueError("SCG requires a finite, ordered sigma window within [0, 1].")
        return io.NodeOutput(KSAMPLER(_make_sampler(profile, strength, sigma_min, sigma_max,
                                                    debug_log, within_bin_mix, budget_trace, spectrum_space)))
