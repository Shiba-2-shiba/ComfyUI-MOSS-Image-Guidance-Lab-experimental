"""Fit SCG Eq. (7) to encoded photos with the installed native MOSS VAE."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--example", type=Path)
    parser.add_argument("--paired-output", type=Path, help="Write measured packed and spatial powers for the same fit images.")
    args = parser.parse_args()
    fit_paths = {p.resolve() for p in args.fit}
    if fit_paths.intersection(p.resolve() for p in args.holdout) or (
        args.example and args.example.resolve() in fit_paths
    ):
        parser.error("Keep evaluation images separate from the fit set.")
    sys.path[:0] = [str(args.comfy.resolve()), str(Path(__file__).resolve().parents[1])]
    import comfy.sd
    import comfy.supported_models
    import comfy.utils
    from scg import FFT_CONVENTION, clean_power, radial_geometry, to_spectral_space

    weights, metadata = comfy.utils.load_torch_file(str(args.vae), return_metadata=True)
    vae = comfy.sd.VAE(sd=weights, metadata=metadata)
    vae.throw_exception_if_invalid()
    latent_format = comfy.supported_models.MOSSImage.latent_format()
    geometry = radial_geometry(64, 64)
    spatial_geometry = radial_geometry(128, 128)
    bins = torch.arange(len(geometry.counts))
    used = (bins > 0) & (geometry.counts > 0)
    freq = bins[used].numpy().astype(np.float64) / 64

    @torch.inference_mode()
    def measure(path, paired=False):
        with Image.open(path) as image:
            rgb = ImageOps.fit(image.convert("RGB"), (1024, 1024), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.array(rgb)).float().div_(255).unsqueeze(0)
        latent = latent_format.process_in(vae.encode(pixels)).cpu()
        if tuple(latent.shape) != (1, 128, 64, 64):
            raise ValueError(f"Expected packed MOSS latent, got {latent.shape}")
        power = clean_power(latent, geometry)
        if paired:
            spatial = clean_power(to_spectral_space(latent, "spatial_measured"), spatial_geometry)
            return power.double().numpy(), spatial.double().numpy()
        return power[used].double().numpy()

    torch.manual_seed(0)
    if args.paired_output:
        pairs = [measure(p, paired=True) for p in args.fit]
        packed, spatial = (np.mean(a, axis=0) for a in zip(*pairs))
        record = dict(fft_convention=FFT_CONVENTION, packed_shape=[128,64,64],
                      vae=args.vae.name, vae_sha256=hashlib.sha256(args.vae.read_bytes()).hexdigest(),
                      photos=[p.name for p in args.fit], encoding_seed=0,
                      packed_measured=packed.tolist(), spatial_measured=spatial.tolist())
        args.paired_output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        train = packed[used.numpy()]
    else:
        train = np.mean([measure(p) for p in args.fit], axis=0)
    valid = np.mean([measure(p) for p in args.holdout], axis=0)

    def curve(params):
        scale, exponent, floor = params
        return scale * freq ** -exponent + floor

    def residual(params):
        return np.log(curve(params)) - np.log(train)

    # Compare initial exponents to reduce sensitivity to the starting point.
    fits = [least_squares(residual, [0.1, slope, 0.1], bounds=([1e-12, 1e-8, 0], np.inf),
                          x_scale="jac") for slope in (0.5, 1., 2., 3.)]
    best = min((fit for fit in fits if fit.success), key=lambda fit: fit.cost)
    print("C, alpha, c:", *best.x)
    # Give the inverse-square baseline its best scale on the same fitting set.
    baseline = np.exp(np.mean(np.log(train) + 2 * np.log(freq))) * freq ** -2

    def report(label, observed):
        for name, predicted in (("inverse-square", baseline), ("Eq7", curve(best.x))):
            error = np.sqrt(np.mean((np.log(predicted) - np.log(observed)) ** 2))
            print(f"{label} {name} log-RMSE: {error:.6f}")

    report("fit", train)
    report("holdout", valid)
    if args.example:
        report("example", measure(args.example))


if __name__ == "__main__":
    main()
