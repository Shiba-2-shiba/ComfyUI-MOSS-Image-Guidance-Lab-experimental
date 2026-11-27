import math

import pytest
import torch


def core():
    try:
        import scg
    except ModuleNotFoundError as error:
        if error.name != "scg":
            raise
        pytest.fail("SCG numerical module is not implemented")
    return scg


def test_spatial_layout_roundtrip_matches_native_flux2_subpixel_order():
    s = core()
    x = torch.arange(2 * 128 * 3 * 5).reshape(2, 128, 3, 5).float()
    u = s.to_spectral_space(x, "spatial_measured")
    assert u.shape == (2, 32, 6, 10)
    for dy in range(2):
        for dx in range(2):
            assert torch.equal(u[:, :, dy::2, dx::2], x[:, 2 * dy + dx::4])
    assert torch.equal(s.from_spectral_space(u, "spatial_measured"), x)
    assert s.to_spectral_space(x, "packed_measured") is x


def test_total_dose_matches_each_sample_and_keeps_spatial_direction():
    s = core()
    x = torch.randn(2, 32, 10, 14, generator=torch.Generator().manual_seed(98))
    geo = s.radial_geometry(10, 14)
    args = (x, s.clean_power(x, geo), geo, .5, 1., .45)
    delta = s.correct(*args) - x
    scales = torch.tensor([.3, .8])
    budget = delta.flatten(1).norm(dim=1) * scales
    adjusted = s.correct(*args, total_l2=budget) - x
    torch.testing.assert_close(adjusted, delta * scales[:, None, None, None], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(adjusted.flatten(1).norm(dim=1), budget)


def test_spatial_fft_separates_low_mode_from_its_packed_alias():
    s = core()
    pos = torch.arange(16) * (2*math.pi/16)
    low = pos.cos().expand(16,16)
    high = (7*pos).cos().expand(16,16)
    u = (low + high)[None,None].repeat(1,32,1,1)
    geom = s.radial_geometry(16,16)
    spectrum = s.clean_power(u,geom)
    spectrum[1] *= 2.25
    spatial = s.correct(u,spectrum,geom,0.,1.,1.)
    torch.testing.assert_close(spatial,(1.5*low+high).expand_as(u),atol=2e-6,rtol=1e-5)
    packed = s.from_spectral_space(u,"spatial_measured")
    coarse = s.radial_geometry(8,8)
    target = s.clean_power(packed,coarse)
    target[1] *= 2.25
    paired = s.to_spectral_space(s.correct(packed,target,coarse,0.,1.,1.),"spatial_measured")
    torch.testing.assert_close(paired,1.5*u,atol=2e-6,rtol=1e-5)


def sine_mode(height, width, axis, phase=0.0):
    pos = torch.arange((height, width)[axis]) * (2 * math.pi / (height, width)[axis])
    if axis == 1:
        return (pos + phase).cos().expand(height, width).clone()[None, None]
    return (pos + phase).cos()[:, None].expand(height, width).clone()[None, None]


@pytest.mark.parametrize("height,width", [(5, 7), (6, 8), (7, 6)])
def test_full_fft_radial_power_matches_independent_loop_and_preserves_modes(height, width):
    scg = core()
    x = 7.0 + sine_mode(height, width, 1, phase=0.7) + 0.4 * sine_mode(height, width, 0, phase=0.2)
    geometry = scg.radial_geometry(height, width)
    actual = scg.radial_power(torch.fft.fft2(x, norm="ortho"), geometry)[0]
    spectrum = torch.fft.fft2(x.double(), norm="ortho")[0, 0]
    powers, counts = {}, {}
    for row in range(height):
        for col in range(width):
            radius = max(height, width) * math.hypot(min(row, height - row) / height, min(col, width - col) / width)
            index = int(radius + .5)
            powers[index] = powers.get(index, 0.0) + abs(spectrum[row, col].item()) ** 2
            counts[index] = counts.get(index, 0) + 1
    want = torch.tensor([powers.get(i, 0.0) / counts.get(i, 1) for i in range(max(counts) + 1)])
    torch.testing.assert_close(actual, want)
    torch.testing.assert_close(geometry.counts, torch.tensor([counts.get(i, 0) for i in range(max(counts) + 1)]))
    clean = scg.clean_power(x, geometry)
    # At sigma=.5 and noise_scale=0 the target is a quarter of source power.
    corrected = scg.correct(x, clean, geometry, .5, noise_scale=0.0, strength=1.0)
    torch.testing.assert_close(corrected, 7.0 + (x - 7.0) * .5, rtol=1e-5, atol=1e-5)


def test_target_uses_flow_coefficients_and_actual_noise_scale():
    torch.testing.assert_close(
        core().target_power(torch.tensor([0.0, 4.0, 16.0]), sigma=.25, noise_scale=2.0),
        torch.tensor([.25, 2.5, 9.25]),
    )


@pytest.mark.parametrize("height,width", [(1, 1), (3, 5), (64, 64)])
def test_analytic_prior_matches_calibrated_curve_in_cycles_per_latent_pixel(height, width):
    scg = core()
    geometry = scg.radial_geometry(height, width)
    power = scg.analytic_power(geometry)
    assert power[0] == 0
    frequency = torch.arange(1, len(power)) / max(height, width)
    torch.testing.assert_close(power[1:], 0.07676908 * frequency ** -1.47746396 + 0.13585077)


def test_correction_keeps_dc_zero_support_and_dtype():
    scg = core()
    x = torch.stack((torch.zeros(6, 8), torch.full((6, 8), 3.0)))[:, None].to(torch.bfloat16)
    geometry = scg.radial_geometry(6, 8)
    result = scg.correct(x, torch.full(geometry.counts.shape, 100.0), geometry, 0.25, 1.0, 1.0)
    assert result.dtype == x.dtype
    assert result.device == x.device
    assert torch.equal(result, x)


def test_correction_preserves_phase_caps_gain_and_does_not_couple_samples():
    scg = core()
    x = sine_mode(8, 8, 1, phase=.7)
    geometry = scg.radial_geometry(8, 8)
    reference = scg.clean_power(x, geometry) * 256.0
    together = scg.correct(torch.cat((x, x * 100.0)), reference, geometry, .01, .0, 1.0)
    alone = scg.correct(x, reference, geometry, .01, .0, 1.0)
    # Exact gains would exceed 2 for the first and be below .5 for the other.
    torch.testing.assert_close(together, torch.cat((x * 2, x * 50)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(together[0:1], alone)


def test_disabled_correction_is_exact_object_passthrough():
    x = torch.tensor([[1.0]])
    assert core().correct(x, None, None, None, None, 0.0) is x


def test_calibration_rejects_nonfinite_latents():
    scg = core()
    x = torch.full((1, 2, 5, 7), torch.nan)
    with pytest.raises(ValueError, match="finite"):
        scg.clean_power(x, scg.radial_geometry(5, 7))


@pytest.mark.parametrize("mix", [.25, 1.0])
def test_within_bin_mix_changes_relative_amplitudes_at_matched_band_energy(mix):
    scg = core()
    x = torch.randn(2, 4, 9, 12, generator=torch.Generator().manual_seed(19))
    x[:, 0] *= 4
    x[:, 1].zero_()
    geo = scg.radial_geometry(9, 12)
    ref = scg.analytic_power(geo)
    fft = lambda v: torch.fft.fft2(v, norm="ortho")
    control = scg.correct(x, ref, geo, .7, 1., .45)
    assert torch.equal(control, scg.correct(x, ref, geo, .7, 1., .45, within_bin_mix=0))
    altered = scg.correct(x, ref, geo, .7, 1., .45, within_bin_mix=mix)
    assert not torch.allclose(altered, control)
    assert torch.equal(altered[:, 1], x[:, 1])
    torch.testing.assert_close(altered.mean((-1,-2)), x.mean((-1,-2)))
    phase = fft(altered) * fft(x).conj()
    assert phase.imag.abs().max() < 1e-4 and phase.real.min() >= -1e-6
    local = scg.radial_power(fft(control-x), geo)
    torch.testing.assert_close(scg.radial_power(fft(altered-x), geo), local, rtol=2e-4, atol=1e-9)
    target = local * .25
    target[:, 0] = 0
    locked = scg.correct(x, ref, geo, .7, 1., .45, within_bin_mix=mix, budget_power=target)
    torch.testing.assert_close(scg.radial_power(fft(locked-x), geo), target, rtol=2e-4, atol=1e-9)
