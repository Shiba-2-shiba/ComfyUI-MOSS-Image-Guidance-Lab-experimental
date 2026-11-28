"""Opt-in pinned ComfyUI CPU smoke; no real VAE/checkpoint or GPU required."""

import importlib
import importlib.util
import csv
import json
from pathlib import Path
import sys
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


def check_two_pass(model, addon, empty):
    from comfy_extras.nodes_custom_sampler import BasicScheduler, CFGGuider, RandomNoise, SamplerCustomAdvanced, SplitSigmas
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    from nodes import KSamplerAdvanced, LatentUpscaleBy

    model, = ModelSamplingAuraFlow().patch_aura(model, 5, "flow")
    positive, negative = [[torch.full((1, 3, 8), .5), {}]], [[torch.zeros(1, 3, 8), {}]]
    guider, = CFGGuider.execute(model, positive, negative, 6.).result
    graph = json.loads((Path(__file__).resolve().parents[1] / "workflows/moss_scg_two_pass.json").read_bytes())
    settings = {n["widgets_values_named"]["sampling_pass"]: n["widgets_values_named"]
                for n in graph["nodes"] if n["type"] == "MossImageSCGEuler"}

    def run_pass(latent, stage, steps, cut, seed, strength, omit_pass=False):
        first = stage == "first_pass"
        if strength is None:
            return KSamplerAdvanced().sample(
                model, "enable", seed, steps, 6., "euler", "simple", positive, negative, latent,
                0 if first else cut, cut if first else steps, "enable" if first else "disable",
            )[0]
        grid, = BasicScheduler.execute(model, "simple", steps, 1.).result
        sigmas = SplitSigmas.execute(grid, cut).result[0 if first else 1]
        inputs = dict(settings[stage], strength=strength)
        if omit_pass:
            del inputs["sampling_pass"]
        sampler, = addon.MossImageSCGEuler.execute(**inputs).result
        noise, = RandomNoise.execute(seed).result
        return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result[0]

    runs = []
    for strength, omit_pass in ((None, False), (0., False), (.45, False), (.45, True)):
        first = run_pass({"samples": empty}, "first_pass", 18, 15, 42, strength, omit_pass)
        enlarged, = LatentUpscaleBy().upscale(first, "bicubic", 1.5)
        final = run_pass(enlarged, "second_pass", 25, 5, 35, strength, omit_pass)
        runs.append((first["samples"], final["samples"]))
    for baseline, disabled, corrected, inferred in zip(*runs):
        assert torch.equal(baseline, disabled)
        assert torch.equal(corrected, inferred)
        assert torch.isfinite(corrected).all() and not torch.equal(baseline, corrected)
    same_latent_control = run_pass(enlarged, "second_pass", 25, 5, 35, None)
    assert not torch.equal(same_latent_control["samples"], runs[-1][1])
    print("Two-pass native MOSS: both outputs at strength 0 matched KSamplerAdvanced Euler; active SCG changed both.")


def main():
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.model_patcher
    import comfy.samplers
    import comfy.sd
    import comfy.supported_models

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    name = "moss_scg_native_smoke"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    addon = importlib.import_module(f"{name}.scg_nodes")
    assert addon.MossImageSCGProfile.define_schema().node_id == "MossImageSCGProfile"
    assert addon.MossImageSCGEuler.define_schema().inputs[0].lazy is True

    cpu = torch.device("cpu")
    cfg = comfy.supported_models.MOSSImage(dict(
        image_model="moss_image", dim=32, ffn_dim=64, num_heads=1, num_layers=1, text_dim=8,
    ))
    cfg.set_inference_dtype(torch.float32, None)
    base = cfg.get_model({}, device=cpu)
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for param in base.diffusion_model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * .05)
    model = comfy.model_patcher.ModelPatcher(base, cpu, cpu)
    vae = object.__new__(comfy.sd.VAE)
    vae.latent_channels = 128
    vae.downscale_ratio = 16
    vae.encode = lambda images: torch.zeros((images.shape[0], 128, 3, 5))
    reference = addon.MossImageSCGProfile.execute(model, vae, torch.zeros(1, 48, 80, 3)).result[0]
    stock = comfy.samplers.ksampler("euler")
    disabled = addon.MossImageSCGEuler.execute(strength=0).result[0]
    active = addon.MossImageSCGEuler.execute(reference, strength=1.0, sigma_min=0, sigma_max=1).result[0]
    analytic = addon.MossImageSCGEuler.execute(strength=.15).result[0]
    noise = torch.randn((1, 128, 3, 5), generator=generator)
    empty = torch.zeros_like(noise)
    sigmas = comfy.samplers.calculate_sigmas(base.model_sampling, "simple", 3)

    def generate(sampler, scale, neg_length):
        count = []
        hook = base.diffusion_model.register_forward_pre_hook(lambda _, args: count.append(args[0].shape[0]))
        try:
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds([[torch.full((1, 3, 8), .5), {}]], [[torch.zeros(1, neg_length, 8), {}]])
            guider.set_cfg(scale)
            return guider.sample(noise, empty, sampler, sigmas, disable_pbar=True, seed=10), count
        finally:
            hook.remove()

    with torch.inference_mode():
        check_two_pass(model, addon, empty)
        for scale, neg_length, expected in ((1.0, 3, [1] * 3), (2.5, 3, [2] * 3), (2.5, 2, [1] * 6)):
            original, original_calls = generate(stock, scale, neg_length)
            bypassed, bypassed_calls = generate(disabled, scale, neg_length)
            corrected, corrected_calls = generate(active, scale, neg_length)
            text_only, text_calls = generate(analytic, scale, neg_length)
            assert torch.equal(original, bypassed)
            assert torch.isfinite(corrected).all() and not torch.equal(original, corrected)
            assert torch.isfinite(text_only).all() and not torch.equal(original, text_only)
            assert original_calls == bypassed_calls == corrected_calls == text_calls == expected
            print(f"SCG native CPU smoke: CFG={scale:g}, negative length={neg_length}, forward batches={expected}.")

        with TemporaryDirectory() as output_dir, patch("folder_paths.get_output_directory", return_value=output_dir):
            for strength in (0, .15):
                traced = addon.MossImageSCGEuler.execute(strength=strength, debug_log="jsonl+latents").result[0]
                sample, calls = generate(traced, 2.5, 2)
                expected_sample, expected_calls = generate(analytic if strength else stock, 2.5, 2)
                assert torch.equal(sample, expected_sample) and calls == expected_calls
            logs = list(Path(output_dir).glob("moss_scg_debug/*/trace.jsonl"))
            assert len(logs) == 2
            runs = {}
            for path in logs:
                records = [json.loads(line) for line in path.read_text().splitlines()]
                assert records[-1]["status"] == "complete" and records[-1]["steps_written"] == 3
                assert records[0]["cfg"] == 2.5 and records[0]["seed"] == 10
                assert records[0]["text_conditions"]["positive"][0]["c_crossattn"]["sha256"]
                runs[records[0]["strength"]] = str(path)
            report = subprocess.check_output(
                [sys.executable, str(root / "scripts/compare_scg_traces.py"), runs[0], runs[.15]], text=True)
            rows = list(csv.DictReader(report.splitlines()))
            assert len(rows) == 3 and float(rows[0]["denoised_ab_relative_l2"]) == 0
            assert float(rows[-1]["post_ab_relative_l2"]) > 0
            controls = set(logs)
            experimental_args = dict(strength=.15, within_bin_mix=.5, budget_trace=Path(runs[.15]).parent.name)
            experiment = addon.MossImageSCGEuler.execute(**experimental_args, debug_log="jsonl+latents").result[0]
            updated, updated_calls = generate(experiment, 2.5, 2)
            plain, plain_calls = generate(addon.MossImageSCGEuler.execute(**experimental_args).result[0], 2.5, 2)
            assert torch.equal(updated, plain) and updated_calls == plain_calls == expected_calls
            experimental_log, = set(Path(output_dir).glob("moss_scg_debug/*/trace.jsonl")) - controls
            comparison = subprocess.check_output(
                [sys.executable, str(root / "scripts/compare_scg_traces.py"), runs[.15], str(experimental_log)], text=True)
            paired = list(csv.DictReader(comparison.splitlines()))
            assert paired[0]["batch_denoised_identical"] == "True"
            applied = [r for r in paired if r["reason"] == "applied"]
            assert applied and all(abs(float(r["scg_l2_ratio"]) - 1.) < 1e-4 for r in applied)
            assert float(paired[-1]["post_ab_relative_l2"]) > 0
            print("Debug traces: native Euler and SCG matched their untraced samples and forward counts.")


if __name__ == "__main__":
    main()
