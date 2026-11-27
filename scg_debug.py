"""Opt-in measurements of the actual MOSS reverse trajectory, one JSON record per line."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
from uuid import uuid4

import torch
from torch.nn.functional import adaptive_avg_pool2d

from .scg import FFT_CONVENTION, _EPSILON, _MAX_AMPLITUDE, _MIN_AMPLITUDE, radial_geometry, radial_power, target_power, to_spectral_space
from .scg_trace_io import MATCH_KEYS, read_run, require_match, require_scg_match


DEBUG_MODES = ("off", "jsonl", "jsonl+latents")


def text_condition_hashes(conditions):
    """Identify native MOSS text conditioning without storing prompts or embedding arrays."""
    result = {}
    for group, items in conditions.items():
        result[group] = []
        for item in items or ():
            tensors = {}
            for name in ("c_crossattn", "context_attention_bias"):
                value = getattr(item.get("model_conds", {}).get(name), "cond", None)
                if isinstance(value, torch.Tensor):
                    data = value.detach().contiguous().cpu()
                    tensors[name] = dict(shape=list(data.shape), dtype=str(data.dtype),
                                         sha256=hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest())
            result[group].append(tensors)
    return result


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ratio(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator, torch.nan)


def _tile_rms(delta):
    spatial = delta.square().mean(dim=1, keepdim=True)
    size = tuple(min(8, d) for d in spatial.shape[-2:])
    return adaptive_avg_pool2d(spatial, size).sqrt()[:, 0]


def _fft(x):
    return torch.fft.fft2(x, norm="ortho")


def _fp32_hash(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def load_budgets(name, metadata, reference, initial):
    """Read per-ring powers or per-sample total L2 from a recorded control."""
    import folder_paths

    root = (Path(folder_paths.get_output_directory()) / "moss_scg_debug").resolve()
    path = (root / name).resolve()
    if path.is_dir():
        path = (path / "trace.jsonl").resolve()
    if not path.is_relative_to(root) or path.name != "trace.jsonl":
        raise ValueError("SCG budget_trace must select a run inside output/moss_scg_debug.")
    _, base, steps = read_run(path)
    if base.get("within_bin_mix", 0) != 0 or not base["strength"]:
        raise ValueError("SCG budget_trace requires a nonzero standard radial SCG control.")
    comparison = dict(metadata, shape=list(initial.shape), dtype=str(initial.dtype), fft_convention=FFT_CONVENTION,
                      initial_sha256_fp32=_fp32_hash(initial.detach().float().cpu()),
                      reference_power=reference.detach().cpu().tolist())
    require_match(base, comparison, MATCH_KEYS)
    require_scg_match(base, comparison)
    total_only = metadata.get("budget_mode") == "total_l2"
    if total_only and base.get("spectrum_space") != "packed_measured":
        raise ValueError("Spatial total-dose control must be a packed_measured run.")
    if not total_only and base.get("spectrum_space", "packed") != metadata.get("spectrum_space", "packed"):
        raise ValueError("Per-ring budgets require identical FFT coordinate systems.")
    if total_only:
        values = [[s["scg_delta_l2"] for s in step["samples"]] for step in steps]
        shape = (len(steps), len(initial))
    else:
        values = [[s["bands"]["power_scg_delta"] for s in step["samples"]] for step in steps]
        shape = (len(steps), len(initial), reference.numel())
    powers = torch.tensor(values, dtype=torch.float32)
    if powers.shape != shape or not torch.isfinite(powers).all() or torch.any(powers < 0):
        raise ValueError("Invalid correction budget in SCG budget_trace.")
    for i, step in enumerate(steps):
        sigma = base["sigmas"][i + 1]
        active = sigma > 0 and base["sigma_min"] <= sigma <= base["sigma_max"]
        if (step["sigma_next"] != sigma or active != (step["reason"] == "applied")
                or any(s.get("fallback_to_euler") for s in step["samples"])):
            raise ValueError(f"Unusable SCG budget_trace at step {i+1}.")
    if not total_only:
        powers[:, :, 0] = 0  # The control has only FFT roundoff in DC.
    return powers.to(device=initial.device), steps[0]["denoised_sha256_fp32"]


class SCGTrace:
    def __init__(self, directory, stream, header, reference, initial):
        self.directory = directory
        self.stream = stream
        self.steps = 0
        self.applied_steps = []
        self.snapshots = header["debug_log"] == "jsonl+latents"
        self.noise_scale = header["noise_scale"]
        self.space = header.get("spectrum_space", "packed")
        shape = to_spectral_space(initial[:1], self.space).shape[1:]
        self.geom = radial_geometry(*shape[-2:])
        self.reference = reference.detach().cpu()
        initial_cpu = initial.detach().float().cpu().contiguous()
        if self.snapshots:
            self.save_states("initial", initial=initial_cpu)
        self.write(dict(
            event="run", schema_version=1, **header, shape=list(initial.shape), dtype=str(initial.dtype),
            device=str(initial.device), torch_version=str(torch.__version__), fft_convention=FFT_CONVENTION,
            initial_sha256_fp32=_fp32_hash(initial_cpu),
            reference_power=self.reference, bin_counts=self.geom.counts, spectral_shape=list(shape),
            frequency_cycles_per_latent_pixel=torch.arange(len(self.geom.counts)) / max(shape[-2:]),
            gain_limits=[_MIN_AMPLITUDE, _MAX_AMPLITUDE], epsilon=_EPSILON,
        ))

    def write(self, record):
        self.stream.write(json.dumps(_json_value(record), allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.flush()

    def save_states(self, name, **states):
        import numpy as np
        np.savez(self.directory / f"{name}.npz", **{k: v.numpy() for k, v in states.items()})

    def step(self, index, sigma, sigma_next, reason, before, denoised, euler, after, audit):
        # Copy only the current step. Debug FFTs and reductions run outside the sampler's device path.
        before, denoised, euler = (v.detach().float().cpu() for v in (before, denoised, euler))
        after = euler if audit is None else after.detach().float().cpu()
        if self.snapshots:
            states = dict(euler=euler, denoised=denoised)
            if audit is not None:
                states["after"] = after
            self.save_states(f"step_{index:04d}", **states)

        denoised_hash, post_hash = _fp32_hash(denoised), _fp32_hash(after)
        before, denoised, euler, after = (to_spectral_space(v, self.space) for v in (before, denoised, euler, after))
        update = euler - before
        correction = after - euler
        flat_update = update.flatten(1)
        flat_correction = correction.flatten(1)
        euler_norm = flat_update.norm(dim=1)
        scg_norm = flat_correction.norm(dim=1)
        spectrum = _fft(euler)
        pre_cpu = radial_power(spectrum, self.geom)
        if audit is None:
            pre = pre_cpu
            target = target_power(self.reference, sigma_next, self.noise_scale)
            applied = torch.ones_like(pre)
        else:
            data = {k: v.detach().cpu() for k, v in audit.items()}
            pre, target = data["source"], data["target"]
            applied = torch.where(data["accepted"][:, None], data["gain"], 1.)
        post = radial_power(_fft(after), self.geom) if audit is not None else pre
        # Total variance = variance between channel means + mean within-channel variance.
        variance = (radial_power(spectrum.square(), self.geom) - pre_cpu.square()).clamp_min(0)
        channel_power = radial_power(spectrum.flatten(0, 1)[:, None], self.geom).reshape(*euler.shape[:2], -1)
        between_channels = channel_power.var(dim=1, unbiased=False)
        update_power = radial_power(_fft(update), self.geom)
        correction_power = radial_power(_fft(correction), self.geom)
        band = dict(
            power_euler=pre, power_post=post, power_denoised=radial_power(_fft(denoised), self.geom),
            power_euler_delta=update_power, power_scg_delta=correction_power,
            scg_over_euler=_ratio(correction_power.sqrt(), update_power.sqrt()),
            euler_power_cv=_ratio(variance.sqrt(), pre_cpu),
            between_channel_power_cv=_ratio(between_channels.sqrt(), pre_cpu),
            within_channel_power_cv=_ratio((variance - between_channels).clamp_min(0).sqrt(), pre_cpu),
            gain_applied=applied,
        )
        eligible = (self.geom.counts > 0) & (torch.arange(len(self.geom.counts)) != 0)
        for name, power in (("pre", pre), ("post", post)):
            band[f"log_error_{name}"] = torch.where(
                eligible, (power + _EPSILON).log() - (target + _EPSILON).log(), torch.nan)
        if audit is not None:
            requested = data["requested"]
            band.update(gain_requested=requested, valid=data["valid"],
                        clamped=data["valid"] & eligible & ((requested < _MIN_AMPLITUDE) | (requested > _MAX_AMPLITUDE)))
            if "budget_power" in data:
                band["radial_gain_control"] = band.pop("gain_applied")
                band["radial_gain_requested"] = band.pop("gain_requested")
                band["radial_clamped"] = band.pop("clamped")
                band["budget_power"] = data["budget_power"]
                band["effective_rms_gain"] = _ratio(post.sqrt(), pre.sqrt())
        sample_data = dict(
            euler_delta_l2=euler_norm, scg_delta_l2=scg_norm,
            scg_over_euler=_ratio(scg_norm, euler_norm),
            scg_over_state=_ratio(scg_norm, euler.flatten(1).norm(dim=1)),
            scg_euler_cosine=_ratio((flat_correction * flat_update).sum(1), scg_norm * euler_norm),
            channel_rms_euler=update.square().mean(dim=(-2, -1)).sqrt(),
            channel_rms_scg=correction.square().mean(dim=(-2, -1)).sqrt(),
            tile_rms_euler=_tile_rms(update), tile_rms_scg=_tile_rms(correction),
            nonfinite_euler=(~torch.isfinite(euler)).flatten(1).sum(1),
            nonfinite_post=(~torch.isfinite(after)).flatten(1).sum(1),
        )
        if audit is not None:
            sample_data["fallback_to_euler"] = ~data["accepted"]
            if "total_l2" in data:
                sample_data.update(requested_total_l2=data["total_l2"], dose_scale=data["dose_scale"])
                band["gain_before_total_match"] = data["unscaled_gain"]
        samples = []
        for b in range(len(before)):
            samples.append(dict(sample=b, **{k: v[b] for k, v in sample_data.items()},
                                bands={k: v[b] for k, v in band.items()}))
        self.write(dict(event="step", step=index, sigma=sigma, sigma_next=sigma_next, reason=reason,
                        denoised_sha256_fp32=denoised_hash, post_sha256_fp32=post_hash,
                        target_power=target, samples=samples))
        self.steps = index
        if reason == "applied":
            self.applied_steps.append(index)


@contextmanager
def open_trace(header, reference, initial):
    import folder_paths

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(folder_paths.get_output_directory()) / "moss_scg_debug" / f"{stamp}_{uuid4().hex}"
    directory.mkdir(parents=True)
    path = directory / "trace.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        logging.info("[MOSS SCG] Debug trace: %s", path)
        trace = SCGTrace(directory, stream, header, reference, initial)
        try:
            yield trace
        except BaseException as exc:
            try:
                trace.write(dict(event="end", status="error", steps_written=trace.steps, error=type(exc).__name__))
            except OSError:
                logging.exception("[MOSS SCG] Could not write the final error record to %s", path)
            raise
        else:
            trace.write(dict(event="end", status="complete", steps_written=trace.steps, applied_steps=trace.applied_steps))
