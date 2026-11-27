"""Opt-in pinned ComfyUI CPU smoke, including the real executor's caching.

Run with the installed v0.46.0 checkout on PYTHONPATH. No model weights,
network, UI, output files or GPU required. The model is genuinely native
MOSS, but small and randomly initialized.
"""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_management
import comfy.model_patcher
from comfy.patcher_extension import WrappersMP
import comfy.samplers
import comfy.supported_models
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced
import execution
import nodes


ROOT = Path(__file__).resolve().parents[1]
NAME = "moss_leap_native_smoke_addon"
spec = importlib.util.spec_from_file_location(NAME, ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
addon = importlib.util.module_from_spec(spec)
sys.modules[NAME] = addon
spec.loader.exec_module(addon)
leap = importlib.import_module(f"{NAME}.leap_nodes")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3210)
    config = comfy.supported_models.MOSSImage({
        "image_model": "moss_image", "dtype": torch.float32, "latent_channels": 128,
        "dim": 64, "ffn_dim": 96, "num_heads": 2, "num_layers": 1, "text_dim": 16,
    })
    cpu = torch.device("cpu")
    base = config.get_model({}, device=cpu)
    with torch.no_grad():
        for param in base.parameters():
            param.normal_(mean=0.0, std=0.1)
    model = comfy.model_patcher.ModelPatcher(base, cpu, cpu)
    positive = [[torch.randn(1, 3, 16), {}]]
    negative = [[torch.randn(1, 2, 16), {}]]
    calibrate_positive = [[torch.randn(1, 4, 16), {}]]
    latent = {"samples": torch.zeros(1, 128, 2, 2), "downscale_ratio_spacial": 16}
    sampling = model.get_model_object("model_sampling")
    grid25 = comfy.samplers.calculate_sigmas(sampling, "simple", 25)
    grid15 = comfy.samplers.calculate_sigmas(sampling, "simple", 15)
    noise = Noise_RandomNoise(2345)
    checkpoint = "tiny-random-native-moss"

    with torch.inference_mode():
        bank_result = leap.MossImageLEAPCalibrate.execute(
            model, calibrate_positive, negative, grid25, latent, 2.5, checkpoint, 44, 2,
        )
        bank, calibration_text = bank_result.result
        assert json.loads(calibration_text)["guided_calls"] == 50
        assert json.loads(calibration_text)["physical_forwards"] == 100
        euler25 = leap.MossImageLEAP.execute(
            model, positive, negative, noise, grid25, latent, "euler", 2.5, checkpoint, 0,
        )
        euler15 = leap.MossImageLEAP.execute(
            model, positive, negative, noise, grid15, latent, "euler", 2.5, checkpoint, 0,
        )
        predicted = leap.MossImageLEAP.execute(
            model, positive, negative, noise, grid25, latent, "leap", 2.5, checkpoint, 0, bank,
        )
        cfg1 = leap.MossImageLEAP.execute(
            model, positive, negative, noise, grid25, latent, "euler", 1.0, checkpoint, 0,
        )
        b1, b2, b3, b4 = [json.loads(result.result[1]) for result in (euler25, euler15, predicted, cfg1)]
        assert [r["guided_calls"] for r in (b1,b2,b3,b4)] == [25,15,15,25]
        assert [r["physical_forwards"] for r in (b1,b2,b3,b4)] == [50,30,30,25]
        assert b1["forward_batch_sizes"] == [1] * 50
        assert {tuple(value) for value in b1["forward_cfg_batches"]} == {(0,), (1,)}
        assert b4["forward_cfg_batches"] == [[0]] * 25
        guide = comfy.samplers.CFGGuider(model)
        guide.set_conds(positive, negative)
        guide.set_cfg(2.5)
        stock = SamplerCustomAdvanced.execute(noise, guide, comfy.samplers.sampler_object("euler"), grid25, latent)
        assert torch.equal(euler25.result[0]["samples"], stock.result[0]["samples"])
        assert torch.isfinite(predicted.result[0]["samples"]).all()
        assert not base.diffusion_model._forward_pre_hooks

    wrapped = model.clone()
    def deny_wrapper(*args, **kwargs):
        raise AssertionError("incompatible wrapper was reached")
    wrapped.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, "fake-lazy-cache", deny_wrapper)
    try:
        leap.MossImageLEAP.execute(wrapped, positive, negative, noise, grid25, latent)
    except ValueError as exc:
        assert "unpatched native model" in str(exc)
    else:
        raise AssertionError("patched ModelPatcher wrapper was accepted")

    # A real Comfy executor reruns only when the sampler input nonce changes.
    # A changed output filename/prefix by itself does not run the sampler.
    terminal_values = []
    class SmokeInputs:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
        RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "NOISE", "SIGMAS", "LATENT", "CONDITIONING")
        FUNCTION = "get"
        def get(self):
            return model, positive, negative, noise, grid25, latent, calibrate_positive

    class SmokeTerminal:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"latent": ("LATENT",), "prefix": ("STRING",)}}
        RETURN_TYPES = ()
        FUNCTION = "save"
        OUTPUT_NODE = True
        def save(self, latent, prefix):
            terminal_values.append((prefix, latent["samples"].clone()))
            return {"ui":{"text":[prefix]}}

    nodes.NODE_CLASS_MAPPINGS.update({
        "SmokeInputs": SmokeInputs, "SmokeTerminal": SmokeTerminal,
        "MossImageLEAP": leap.MossImageLEAP, "MossImageLEAPCalibrate": leap.MossImageLEAPCalibrate,
    })
    server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *args: None)
    executor = execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC,
                                       cache_args={"ram": 0.0, "ram_inactive": 0.0})
    prompt = {
        "1":{"class_type":"SmokeInputs","inputs":{}},
        "2":{"class_type":"MossImageLEAP","inputs":{
            **{name:["1", index] for index,name in enumerate(
                ("model", "positive", "negative", "noise", "sigmas", "latent_image"))},
            "mode":"euler", "cfg":2.5, "checkpoint_id":checkpoint, "run_nonce":100,
        }},
        "3":{"class_type":"SmokeTerminal","inputs":{"latent":["2",0],"prefix":"first"}},
    }
    def execute(active_prompt, job):
        executor.execute(active_prompt, job, execute_outputs=["3"])
        assert executor.success, executor.status_messages
        cache = [m["nodes"] for ev,m in executor.status_messages if ev == "execution_cached"][0]
        return executor.history_result, cache

    first, cache = execute(prompt, "job-a")
    assert "2" not in cache
    first_report = json.loads(first["outputs"]["2"]["text"][0])
    assert first_report["run_nonce"] == 100 and first_report["physical_forwards"] == 50
    second_prompt = copy.deepcopy(prompt)
    second_prompt["3"]["inputs"]["prefix"] = "changed-only-prefix"
    second, cache = execute(second_prompt, "job-b")
    assert "2" in cache
    assert "2" not in second["outputs"]  # fresh history has no report for the cached sampler
    third_prompt = copy.deepcopy(second_prompt)
    third_prompt["2"]["inputs"]["run_nonce"] = 101
    third, cache = execute(third_prompt, "job-c")
    assert "1" in cache and "2" not in cache
    third_report = json.loads(third["outputs"]["2"]["text"][0])
    assert third_report["run_nonce"] == 101 and third_report["physical_forwards"] == 50
    assert len(terminal_values) == 3 and all(
        torch.equal(terminal_values[0][1], x[1]) for x in terminal_values[1:])
    leap_prompt = copy.deepcopy(third_prompt)
    leap_prompt["4"] = {
        "class_type": "MossImageLEAPCalibrate", "inputs": {
            "model": ["1", 0], "positive": ["1", 6], "negative": ["1", 2],
            "sigmas": ["1", 4], "latent_image": ["1", 5], "cfg": 2.5,
            "checkpoint_id": checkpoint, "calibration_seed": 44, "reference_count": 2,
        },
    }
    leap_prompt["2"]["inputs"].update(mode="leap", bank=["4", 0], run_nonce=102)
    fourth, cache = execute(leap_prompt, "job-d")
    assert "4" not in cache
    assert json.loads(fourth["outputs"]["4"]["text"][0])["guided_calls"] == 50
    assert json.loads(fourth["outputs"]["2"]["text"][0])["physical_forwards"] == 30
    leap_prompt["2"]["inputs"]["run_nonce"] = 103
    fifth, cache = execute(leap_prompt, "job-e")
    assert "4" in cache and "2" not in cache
    assert json.loads(fifth["outputs"]["2"]["text"][0])["physical_forwards"] == 30
    assert torch.equal(terminal_values[-1][1], terminal_values[-2][1])
    print("Native ComfyUI LEAP CPU smoke PASS: guided 25/15/15, physical 50/30/30 "
          "(CFG 2.5, split), CFG1 physical 25, stock Euler bitwise, cache/nonce/history and reuse of calibration bank validated.")


if __name__ == "__main__":
    main()
