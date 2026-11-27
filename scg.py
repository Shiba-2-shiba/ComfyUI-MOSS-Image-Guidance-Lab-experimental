"""Spectral Correction Guidance on normalized MOSS-Image latent states."""

from dataclasses import dataclass

import torch


FFT_CONVENTION = "torch-fft2-ortho-fp32-round-radial-long-axis-v1"
_EPSILON = 1e-12
_MIN_AMPLITUDE = 0.5
_MAX_AMPLITUDE = 2.0
SPECTRUM_SPACES = ("packed", "packed_measured", "spatial_measured")


def to_spectral_space(x, space):
    """Undo only the FLUX.2 VAE's 2x2 permutation, preserving normalized flow coordinates."""
    return torch.nn.functional.pixel_shuffle(x, 2) if space == "spatial_measured" else x


def from_spectral_space(x, space):
    return torch.nn.functional.pixel_unshuffle(x, 2) if space == "spatial_measured" else x


@dataclass(frozen=True)
class RadialGeometry:
    height: int
    width: int
    bins: torch.Tensor
    counts: torch.Tensor


def radial_geometry(height: int, width: int, device=None) -> RadialGeometry:
    """Bin by nearest frequency in units of one cycle per longest axis."""
    if height < 1 or width < 1:
        raise ValueError("SCG requires a nonempty spatial grid.")
    fy = torch.fft.fftfreq(height, dtype=torch.float64, device="cpu")[:, None]
    fx = torch.fft.fftfreq(width, dtype=torch.float64, device="cpu")[None, :]
    radius = (fy.square() + fx.square()).sqrt() * max(height, width)
    bins = (radius + .5).floor().flatten().long()
    counts = torch.bincount(bins)
    return RadialGeometry(height, width, bins.to(device=device), counts.to(device=device))


def radial_power(spectrum: torch.Tensor, geometry: RadialGeometry) -> torch.Tensor:
    """Return one mean power per sample and bin, pooling channels equally."""
    if spectrum.ndim != 4 or spectrum.shape[-2:] != (geometry.height, geometry.width):
        raise ValueError("SCG spectrum does not match the NCHW radial geometry.")
    return _radial_mean(spectrum.real.square() + spectrum.imag.square(), geometry)


def _radial_mean(values, geometry):
    per_frequency = values.mean(dim=1).flatten(1)
    bin_ids = geometry.bins.expand(values.shape[0], -1)
    totals = per_frequency.new_zeros((values.shape[0], geometry.counts.numel()))
    totals.scatter_add_(1, bin_ids, per_frequency)
    return totals / geometry.counts.clamp_min(1)


def clean_power(latent: torch.Tensor, geometry: RadialGeometry) -> torch.Tensor:
    if latent.ndim != 4 or min(latent.shape) == 0 or not latent.is_floating_point():
        raise ValueError("SCG requires a nonempty floating point NCHW reference latent.")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("SCG reference latent must be finite.")
    spectrum = torch.fft.fft2(latent.float(), norm="ortho")
    power = radial_power(spectrum, geometry).mean(dim=0)
    if not bool(torch.isfinite(power).all()):
        raise ValueError("SCG reference power must be finite in FP32.")
    return power


def analytic_power(geometry: RadialGeometry) -> torch.Tensor:
    """Eq. (7), fitted in cycles per packed latent pixel; DC is preserved by correct()."""
    power = torch.zeros_like(geometry.counts, dtype=torch.float32)
    frequency = torch.arange(1, power.numel(), device=power.device, dtype=power.dtype)
    frequency /= max(geometry.height, geometry.width)
    power[1:] = 0.07676908 * frequency.pow(-1.47746396) + 0.13585077
    return power


def target_power(reference: torch.Tensor, sigma: float, noise_scale: float) -> torch.Tensor:
    """Expected native CONST-flow power with orthonormal FFT and white noise."""
    return reference * ((1.0 - sigma) ** 2) + (sigma * noise_scale) ** 2


def _redistribute(spectrum, source, gains, geometry, mix, budget_power):
    """Turn the amplitude update toward a common magnitude, at fixed RMS per sample and ring."""
    def expand(bands):
        return bands[:, geometry.bins].reshape(-1, 1, geometry.height, geometry.width)

    amplitude = spectrum.abs()
    support = amplitude > 0
    density = _radial_mean(support.float(), geometry)
    # Flat on existing support only. Both the flat and radial endpoints have RMS g*sqrt(S).
    safe_density = torch.where(density > 0, density, 1.)
    flat = torch.where(support, expand((source / safe_density).sqrt() * gains), 0.)
    radial = amplitude * expand(gains)
    budget = source.sqrt() * (gains - 1).abs() if budget_power is None else budget_power.sqrt()
    flat_rms = _radial_mean((flat - amplitude).square(), geometry).sqrt()
    # Blend directions with equal nominal length so a tiny radial dose is still adjustable.
    denominator = (1 - mix) * flat_rms + mix * budget
    weight = mix * budget / torch.where(denominator > 0, denominator, 1.)
    shape = torch.lerp(radial, flat, expand(weight))
    shape_rms = _radial_mean(shape.square(), geometry).sqrt()
    shell = source.sqrt() * gains / torch.where(shape_rms > 0, shape_rms, 1.)
    direction = shape * expand(shell) - amplitude
    rms = _radial_mean(direction.square(), geometry).sqrt()
    scale = budget / torch.where(rms > 0, rms, 1.)
    if budget_power is None:
        # Reverse triangle inequality: the nearest point on the target shell is the radial endpoint.
        scale = scale.clamp(max=1.)
    delta = direction * expand(scale)
    if budget_power is not None:
        if bool(((budget > 0) & (rms == 0)).any()) or bool((amplitude + delta < 0).any()):
            raise ValueError("SCG recorded budget is incompatible with this phase-preserving update.")
    delta = torch.maximum(delta, -amplitude)
    return torch.where(expand(budget > 0), torch.sgn(spectrum) * delta, 0.), budget.square()


def correct(
    latent: torch.Tensor,
    reference: torch.Tensor,
    geometry: RadialGeometry,
    sigma: float,
    noise_scale: float,
    strength: float,
    audit: dict | None = None,
    *, within_bin_mix: float = 0., budget_power: torch.Tensor | None = None,
    total_l2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply radial SCG, optionally redistributing its correction energy within each bin."""
    if strength == 0.0:
        return latent
    spectrum = torch.fft.fft2(latent.float(), norm="ortho")
    source = radial_power(spectrum, geometry)
    target = target_power(reference, sigma, noise_scale)
    # The half exponent converts the power ratio to an amplitude ratio.
    log_gain = .5 * strength * ((target + _EPSILON).log() - (source + _EPSILON).log())
    requested = log_gain.exp()
    gains = requested.clamp(_MIN_AMPLITUDE, _MAX_AMPLITUDE)
    valid = (source > 0) & torch.isfinite(source) & torch.isfinite(target) & (geometry.counts > 0)
    gains = torch.where(valid, gains, 1.0)
    # Only bin zero contains DC. Avoid changing the latent channel means.
    gains[:, 0] = 1.0
    if total_l2 is not None:
        if within_bin_mix or budget_power is not None:
            raise ValueError("Total-dose comparison requires a radial update.")
        # Parseval with ortho FFT. source averages over channels and coefficients in each ring.
        relative_gain = gains - 1
        norm = ((source * relative_gain.square() * geometry.counts).sum(1) * latent.shape[1]).sqrt()
        scale = total_l2 / torch.where(norm > 0, norm, 1.)
        scaled = relative_gain * scale[:, None]
        if bool(((norm == 0) & (total_l2 > 0)).any()) or bool((scaled < -1).any()):
            raise ValueError("Recorded total dose cannot be applied along this phase-preserving spatial direction.")
        pixel_delta = scaled[:, geometry.bins].reshape(-1, 1, geometry.height, geometry.width)
        output = (latent.float() + torch.fft.ifft2(spectrum * pixel_delta, norm="ortho").real).to(latent.dtype)
        if audit is not None:
            audit.update(unscaled_gain=gains, total_l2=total_l2, dose_scale=scale)
        gains = 1 + scaled
    elif within_bin_mix:
        delta, budget = _redistribute(spectrum, source, gains, geometry, within_bin_mix, budget_power)
        # Transform only the small change, avoiding a full-state FFT round trip for zero-dose rings.
        output = (latent.float() + torch.fft.ifft2(delta, norm="ortho").real).to(dtype=latent.dtype)
        if audit is not None:
            audit["budget_power"] = budget
    else:
        pixel_gain = gains[:, geometry.bins].reshape(-1, 1, geometry.height, geometry.width)
        output = torch.fft.ifft2(spectrum * pixel_gain, norm="ortho").real.to(dtype=latent.dtype)
    # Extreme numerical overflow must not inject nonfinite values into a good sample.
    finite = torch.isfinite(output).flatten(1).all(dim=1).view(-1, 1, 1, 1)
    if (budget_power is not None or total_l2 is not None) and not bool(finite.all()):
        raise ValueError("SCG recorded budget produced a nonfinite update.")
    if audit is not None:
        audit.update(source=source, target=target, requested=requested, gain=gains,
                     valid=valid, accepted=finite.flatten())
    return torch.where(finite, output, latent)
