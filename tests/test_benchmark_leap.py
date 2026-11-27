import copy
import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bench():
    return runpy.run_path(str(ROOT / "scripts/benchmark_leap.py"))


@pytest.fixture
def graph():
    ui = json.loads((ROOT / "workflows/moss_leap_compare.json").read_text(encoding="utf-8"))
    edges = {row[0]: [str(row[1]), row[2]] for row in ui["links"]}
    result = {str(node["id"]): {"class_type": node["type"], "inputs": {
        inp["name"]: edges[inp["link"]] for inp in node["inputs"]
    }} for node in ui["nodes"]}
    result["5"]["inputs"]["noise_seed"] = 438829726369282
    for key, steps in (("9", 25), ("10", 15)):
        result[key]["inputs"].update(scheduler="simple", steps=steps, denoise=1)
    for key in ("16", "17", "18"):
        result[key]["inputs"]["mode"] = "leap" if key == "18" else "euler"
    result["15"]["inputs"].update(calibration_seed=1000, reference_count=2)
    result["6"]["inputs"]["text"] = "test render"
    result["11"]["inputs"]["text"] = "calibration render"
    return result


def fake_transport(*, invalid=None):
    submissions = []
    def request(method, path, data=None, *, timeout):
        assert timeout > 0
        if method == "GET" and path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if method == "POST" and path == "/prompt":
            submissions.append(copy.deepcopy(data))
            return {"prompt_id": "00000000-0000-4000-8000-000000000001"}
        assert method == "GET" and path == "/history/00000000-0000-4000-8000-000000000001"
        arm, = submissions[-1]["partial_execution_targets"]
        sid, steps, calls = {"23": ("16", 25, 25), "25": ("17", 15, 15), "27": ("18", 25, 15)}[arm]
        nonce = submissions[-1]["prompt"][sid]["inputs"]["run_nonce"]
        cached = ["15"] if arm == "27" and len(submissions) > 3 else []
        if invalid == "sampler cached":
            cached += [sid]
        if invalid == "calibration rerun":
            cached = []
        report = dict(schema_version=1, mode="leap" if arm == "27" else "euler",
            run_nonce=nonce if invalid != "stale nonce" else nonce-1, seed=438829726369282,
            steps=steps, guided_calls=calls if invalid != "bad count" else 2,
            physical_forwards=calls if invalid != "no forwards" else 0,
            forward_batch_sizes=[2]*calls, forward_cfg_batches=[[0,1]]*calls,
            sampler_seconds=1.25, device="cuda:0", gpu_name="fake",
            torch_version="mock", torch_hip="mock", torch_cuda=None,
            sigmas=[1 - i/steps for i in range(steps+1)])
        history = {"status": {"status_str": "success", "completed": True, "messages": [
            ["execution_cached", {"nodes": cached, "prompt_id": path.rsplit("/",1)[-1]}],
        ]}, "outputs": {
            sid: {"text": [json.dumps(report)]},
            arm: {"images": [{"filename": "output.png", "type": "output", "subfolder": ""}]},
        }}
        if invalid == "other terminal":
            other = "25" if arm == "23" else "23"
            history["outputs"][other] = {"images": [{"filename": "unexpected.png"}]}
        return {path.rsplit("/",1)[-1]: history}
    return request, submissions


def test_rotates_after_excluded_warmup_and_changes_only_selected_nonce(bench, graph, tmp_path):
    request, submissions = fake_transport()
    result = bench["run_benchmark"](graph, request=request, repeats=3, timeout=3,
                                    output=tmp_path/"benchmark.json")
    assert [trial["partial_execution_targets"] for trial in submissions] == [
        ["23"], ["25"], ["27"], ["23"], ["25"], ["27"], ["25"], ["27"], ["23"], ["27"], ["23"], ["25"],
    ]
    nonces = set()
    for submission in submissions:
        selected, = submission["partial_execution_targets"]
        node = {"23":"16", "25":"17", "27":"18"}[selected]
        nonce = submission["prompt"][node]["inputs"]["run_nonce"]
        assert type(nonce) is int and nonce not in nonces
        nonces.add(nonce)
        submission["prompt"][node]["inputs"]["run_nonce"] = graph[node]["inputs"]["run_nonce"]
        assert submission["prompt"] == graph
    assert len(result["warmups"]) == 3 and len(result["trials"]) == 9
    assert result["summary"]["leap25_15"]["median_sampler_seconds"] == 1.25
    assert "queue_to_history_seconds" in result["trials"][0]
    assert json.loads((tmp_path/"benchmark.json").read_text()) == result


@pytest.mark.parametrize("invalid", ["sampler cached", "stale nonce", "bad count", "no forwards"])
def test_rejects_cached_or_nonfresh_sampler_evidence(bench, graph, tmp_path, invalid):
    request, submissions = fake_transport(invalid=invalid)
    out = tmp_path/"benchmark.json"
    with pytest.raises(ValueError):
        bench["run_benchmark"](graph, request=request, repeats=1, timeout=3, output=out)
    assert len(submissions) == 1
    assert not out.exists()


def test_does_not_report_warm_leap_if_calibration_ran_again(bench, graph, tmp_path):
    request, _ = fake_transport(invalid="calibration rerun")
    out = tmp_path/"benchmark.json"
    with pytest.raises(ValueError, match="calibration"):
        bench["run_benchmark"](graph, request=request, repeats=1, timeout=3, output=out)
    assert len(json.loads(out.read_text())["trials"]) == 2


def test_different_noise_fails_before_post(bench, graph, tmp_path):
    request, submissions = fake_transport()
    graph["18"]["inputs"]["noise"] = ["14", 0]
    with pytest.raises(ValueError, match="noise"):
        bench["run_benchmark"](graph, request=request, repeats=1, timeout=3,
                               output=tmp_path/"benchmark.json")
    assert not submissions


def test_timing_out_does_not_enqueue_replacement(bench, graph, tmp_path):
    posts = []
    def request(method, path, data=None, *, timeout):
        if path == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if method == "POST":
            posts.append(data)
            return {"prompt_id": "00000000-0000-4000-8000-000000000001"}
        return {}
    with pytest.raises(TimeoutError, match="may still"):
        bench["run_benchmark"](graph, request=request, repeats=1, timeout=0.02,
                               output=tmp_path/"benchmark.json")
    assert len(posts) == 1


def test_rejects_backend_that_ignores_single_terminal_selection(bench, graph, tmp_path):
    request, submissions = fake_transport(invalid="other terminal")
    out = tmp_path / "benchmark.json"
    with pytest.raises(ValueError, match="unexpected SaveImage"):
        bench["run_benchmark"](graph, request=request, repeats=1, timeout=3, output=out)
    assert len(submissions) == 1
    assert not out.exists()
