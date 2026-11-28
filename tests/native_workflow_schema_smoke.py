"""Validate the editable workflow's sockets against actual ComfyUI node metadata.

From this repository, put the installed ComfyUI checkout on PYTHONPATH and run
python tests/native_workflow_schema_smoke.py. No checkpoints or GPU are used.
"""

import asyncio
import json
from pathlib import Path

import comfy.cli_args

comfy.cli_args.args.cpu = True
import nodes
from comfy_api.latest._io import _ComfyNodeInternal, get_finalized_class_inputs

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(nodes.__file__).resolve().parent


def info(cls, node):
    if issubclass(cls, _ComfyNodeInternal):
        data = cls.GET_NODE_INFO_V1()
        live_inputs = dict(node.get("widgets_values_named", {}))
        live_inputs.update({p["name"]: p["link"] for p in node["inputs"]})
        inputs, _, _ = get_finalized_class_inputs(cls.INPUT_TYPES(), live_inputs)
        return inputs, data["output"]
    return cls.INPUT_TYPES(), cls.RETURN_TYPES


async def main():
    extras = ("nodes_custom_sampler", "nodes_flux", "nodes_primitive", "nodes_text", "nodes_lora_stack",
              "nodes_resolution", "nodes_math", "nodes_model_advanced")
    for name in extras:
        assert await nodes.load_custom_node(str(COMFY / "comfy_extras" / f"{name}.py"),
                                            module_parent="comfy_extras"), name
    assert await nodes.load_custom_node(str(ROOT.parent / "AFM_for_MossImage_experimental")), "BlockTune registration failed"
    before = set(nodes.NODE_CLASS_MAPPINGS)
    assert await nodes.load_custom_node(str(ROOT)), "extension registration failed"
    assert set(nodes.NODE_CLASS_MAPPINGS) - before == {
        "MossImageLEAP", "MossImageLEAPCalibrate", "MossImageSCGEuler", "MossImageSCGProfile",
    }, "expected only SCG and LEAP node registration"
    for path in sorted((ROOT / "workflows").glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        for n in graph["nodes"]:
            if n["type"] == "MarkdownNote":  # Frontend-only annotation.
                assert not n["inputs"] and not n["outputs"]
                continue
            cls = nodes.NODE_CLASS_MAPPINGS[n["type"]]
            inputs, outputs = info(cls, n)
            sockets = dict(inputs.get("required", {})) | dict(inputs.get("optional", {}))
            for port in n["inputs"]:
                assert port["name"] in sockets, (path.name, n["id"], port)
                assert sockets[port["name"]][0] == port["type"], (path.name, n["id"], port)
            assert tuple(o["type"] for o in n["outputs"]) == tuple(outputs[:len(n["outputs"])]), (path.name, n["id"], outputs)
        print(f"{path.name}: {len(graph['nodes'])} native and extension node schemas matched")


if __name__ == "__main__":
    asyncio.run(main())
