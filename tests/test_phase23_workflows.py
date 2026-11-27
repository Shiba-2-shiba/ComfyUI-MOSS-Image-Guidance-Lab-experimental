import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    graph = json.loads((ROOT / "workflows" / name).read_text("utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {item[0]: item for item in graph["links"]}
    assert len(nodes) == len(graph["nodes"])
    assert len(links) == len(graph["links"])
    assert max(nodes) == graph["last_node_id"]
    assert max(links) == graph["last_link_id"]
    for lid, src, out, dst, inp, kind in links.values():
        assert nodes[src]["order"] < nodes[dst]["order"]
        assert lid in nodes[src]["outputs"][out]["links"]
        assert kind == nodes[src]["outputs"][out]["type"]
        assert lid == nodes[dst]["inputs"][inp]["link"]
        assert kind == nodes[dst]["inputs"][inp]["type"]
    for node in nodes.values():
        for inp in node["inputs"]:
            assert inp["link"] is None or inp["link"] in links
        for out in node["outputs"]:
            assert set(out["links"] or []) <= links.keys()
    return nodes, lambda n: {i["name"]: links[i["link"]][1] for i in n["inputs"] if i["link"] is not None}


def test_scg_pair_has_shared_sampling_inputs_except_sampler():
    nodes, sources = load("moss_scg_ab.json")
    base, scg = [n for n in nodes.values() if n["type"] == "SamplerCustomAdvanced"]
    a, b = sources(base), sources(scg)
    assert nodes[a.pop("sampler")]["widgets_values"] == ["euler"]
    corrected = nodes[b.pop("sampler")]
    assert corrected["type"] == "MossImageSCGEuler"
    assert corrected["widgets_values"] == [0.15, 0.1, 0.95, "off", 0.0, "", "packed"]
    assert a == b
    assert "profile" not in sources(corrected)
    assert all(n["type"] != "LoadImage" for n in nodes.values())
    assert nodes[a["latent_image"]]["type"] == "EmptyFlux2LatentImage"
    tuned = nodes[sources(nodes[a["guider"]])["model"]]
    assert sources(nodes[a["sigmas"]])["model"] == tuned["id"]
    assert tuned["type"] == "MossImageBlockTunePreset" and tuned["mode"] == 0
    assert tuned["widgets_values"] == ["image_3_individual_qkv_partial", "static"]
    lora = nodes[sources(tuned)["model"]]
    assert lora["type"] == "LoadLoraModel" and lora["mode"] == 0
    assert lora["widgets_values"] == [1, r"MoodyMoss\realistic_femaleonly_deaging.safetensors", 1, True]
    assert nodes[sources(lora)["model"]]["type"] == "UNETLoader"
    assert len({sources(n)["vae"] for n in nodes.values() if n["type"] == "VAEDecode"}) == 1


def test_leap_triplet_has_matched_noise_cfg_and_grid():
    nodes, sources = load("moss_leap_compare.json")
    e25, e15, leap = [n for n in nodes.values() if n["type"] == "MossImageLEAP"]
    assert [n["widgets_values"][0] for n in [e25, e15, leap]] == ["euler", "euler", "leap"]
    p, q, r = [sources(n) for n in [e25, e15, leap]]
    calibration = nodes[r.pop("bank")]
    assert calibration["type"] == "MossImageLEAPCalibrate"
    full, short, full_again = [s.pop("sigmas") for s in [p, q, r]]
    assert full == full_again == sources(calibration)["sigmas"]
    assert nodes[full]["widgets_values"] == ["simple", 25, 1.0]
    assert nodes[short]["widgets_values"] == ["simple", 15, 1.0]
    assert p == q == r
    cs = sources(calibration)
    assert (cs["positive"], cs["negative"]) != (p["positive"], p["negative"])
    assert {key: cs[key] for key in ("model", "latent_image", "cfg")} == {key: p[key] for key in ("model", "latent_image", "cfg")}
    assert [n["id"] for n in nodes.values() if n["type"] == "SaveImage"] == [23, 25, 27]
