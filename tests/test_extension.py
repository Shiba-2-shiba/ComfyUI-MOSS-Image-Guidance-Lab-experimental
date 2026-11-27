import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_public_entrypoint_registers_only_scg_and_leap_nodes(monkeypatch):
    # Stub the ComfyUI schema boundary, but load the real extension and node classes.
    comfy_api = ModuleType("comfy_api")
    comfy_api.__path__ = []
    latest = ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})

    def port(name=None, **kwargs):
        return SimpleNamespace(name=name, **kwargs)

    port_type = SimpleNamespace(Input=port, Output=port)
    latest.io = SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}), Input=port, Output=port,
        Schema=lambda **kwargs: SimpleNamespace(**kwargs), Custom=lambda _: port_type,
        NodeOutput=lambda *values: SimpleNamespace(result=values),
        **{name: port_type for name in (
            "Boolean", "Combo", "Conditioning", "Float", "Image", "Int", "Latent",
            "Model", "Noise", "Sampler", "Sigmas", "String", "Vae",
        )},
    )
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    spec = importlib.util.spec_from_file_location(
        "_moss_guidance_registry_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
    )
    extension_package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, extension_package)
    try:
        spec.loader.exec_module(extension_package)
        extension = asyncio.run(extension_package.comfy_entrypoint())
        actual = asyncio.run(extension.get_node_list())
        assert sorted(cls.define_schema().node_id for cls in actual) == [
            "MossImageLEAP", "MossImageLEAPCalibrate", "MossImageSCGEuler", "MossImageSCGProfile",
        ]
    finally:
        for name in list(sys.modules):
            if name.startswith(spec.name + "."):
                sys.modules.pop(name)


def test_distributed_editable_workflows_are_only_scg_and_leap():
    assert sorted(path.name for path in (ROOT / "workflows").glob("*.json")) == [
        "moss_leap_compare.json", "moss_scg_ab.json",
    ]
