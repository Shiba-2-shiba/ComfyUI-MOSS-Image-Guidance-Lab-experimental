#!/usr/bin/env python3
"""Run the supplied MOSS-Image LEAP API graph on idle local ComfyUI with its normal cache enabled."""

import argparse
import copy
import http.client
import json
import math
from pathlib import Path
import secrets
import statistics
import sys
import time
import uuid


# Label, sampler node, SaveImage node, mode, Euler updates, guided requests
ARMS = (
    ("euler25", "16", "23", "euler", 25, 25),
    ("euler15", "17", "25", "euler", 15, 15),
    ("leap25_15", "18", "27", "leap", 25, 15),
)
CALIBRATION = "15"
POLL_SECONDS = 0.25


def _inputs(graph, node, kind):
    entry = graph.get(node)
    if not isinstance(entry, dict) or entry.get("class_type") != kind or not isinstance(entry.get("inputs"), dict):
        raise ValueError(f"Expected node {node} ({kind}); export the supplied LEAP workflow in API format.")
    return entry["inputs"]


def _node_link(value):
    if (not isinstance(value, list) or len(value) != 2 or
            not isinstance(value[0], str) or value[1] != 0):
        raise ValueError("Expected a link to a node's first output in the supplied API graph.")
    return value[0]


def validate_graph(graph):
    if not isinstance(graph, dict):
        raise ValueError("Expected the supplied LEAP workflow exported in API format.")
    samplers = []
    vaes = []
    schedules = []
    for _, sid, terminal, mode, steps, _ in ARMS:
        sampler = _inputs(graph, sid, "MossImageLEAP")
        decoder_id = _node_link(_inputs(graph, terminal, "SaveImage").get("images"))
        decoder = _inputs(graph, decoder_id, "VAEDecode")
        if decoder.get("samples") != [sid, 0]:
            raise ValueError(f"SaveImage {terminal} does not decode sampler {sid}.")
        vaes.append(_node_link(decoder.get("vae")))
        schedule_id = _node_link(sampler.get("sigmas"))
        schedule = _inputs(graph, schedule_id, "BasicScheduler")
        if (sampler.get("mode") != mode or schedule.get("steps") != steps
                or schedule.get("denoise") != 1 or schedule.get("model") != sampler.get("model")):
            raise ValueError(f"Node {sid} does not match its full-denoise {mode} / {steps} scope.")
        schedules.append(schedule)
        samplers.append(sampler)
    for key in ("model", "positive", "negative", "latent_image", "noise", "cfg", "checkpoint_id"):
        left = samplers[0].get(key)
        if left is None or any(s.get(key) != left for s in samplers[1:]):
            raise ValueError(f"All three sampler nodes must share {key}.")
    if (samplers[0]["sigmas"] != samplers[2]["sigmas"] or
            any(s.get("scheduler") != schedules[0].get("scheduler") for s in schedules) or
            len(set(vaes)) != 1):
        raise ValueError("Match the 25-update grids, scheduler type and VAE across the three arms.")
    if any(s.get("bank") is not None for s in samplers[:2]) or samplers[2].get("bank") != [CALIBRATION, 0]:
        raise ValueError("Only the LEAP arm must use calibration node 15.")
    calibration = _inputs(graph, CALIBRATION, "MossImageLEAPCalibrate")
    for key in ("model", "latent_image", "cfg", "checkpoint_id", "sigmas"):
        if calibration.get(key) != samplers[2].get(key):
            raise ValueError(f"LEAP calibration and evaluation must share {key}.")

    noise_id = _node_link(samplers[0]["noise"])
    seed = _inputs(graph, noise_id, "RandomNoise").get("noise_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("Set the evaluation RandomNoise to a fixed literal integer seed.")
    calibration_seed, refs = calibration.get("calibration_seed"), calibration.get("reference_count")
    if (type(calibration_seed) is not int or type(refs) is not int or refs < 1 or
            calibration_seed <= seed < calibration_seed + refs):
        raise ValueError("Use a literal calibration seed/count and an evaluation seed outside that range.")
    eval_pos = _inputs(graph, _node_link(samplers[0]["positive"]), "CLIPTextEncode")
    calib_pos = _inputs(graph, _node_link(calibration.get("positive")), "CLIPTextEncode")
    if (not isinstance(eval_pos.get("text"), str) or not isinstance(calib_pos.get("text"), str) or
            not eval_pos["text"].strip() or not calib_pos["text"].strip() or
            eval_pos["text"].strip() == calib_pos["text"].strip()):
        raise ValueError("Set distinct nonempty evaluation and calibration positive prompts.")
    return seed


def local_request(port):
    """Use a fixed loopback host, with neither proxy nor redirect support."""
    def request(method, path, data=None, *, timeout):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            body = json.dumps(data).encode("utf-8") if data is not None else None
            connection.request(method, path, body, {"Content-Type": "application/json"} if body else {})
            response = connection.getresponse()
            payload = response.read(10 * 1024 * 1024 + 1)
            if response.status != 200:
                raise ValueError(f"Local ComfyUI {method} {path}: HTTP {response.status}: "
                                 f"{payload[:1000].decode('utf-8', 'replace')}")
            if len(payload) > 10 * 1024 * 1024:
                raise ValueError("Local ComfyUI response exceeds 10 MiB.")
            return json.loads(payload)
        finally:
            connection.close()
    return request


def _fresh_report(item, prompt_id, arm, nonce, seed, warmup):
    _, sampler, terminal, mode, steps, guided = arm
    status = item.get("status", {})
    messages = status.get("messages", ())
    if status.get("status_str") != "success" or status.get("completed") is not True:
        details = [m for m in messages if m[0] in ("execution_error", "execution_interrupted")]
        raise ValueError(f"ComfyUI prompt {prompt_id} did not complete successfully: {str(details)[:1000]}")
    cache_events = [p for event, p in messages if event == "execution_cached" and p.get("prompt_id") == prompt_id]
    if len(cache_events) != 1 or not isinstance(cache_events[0].get("nodes"), list):
        raise ValueError("Missing execution_cached audit for this prompt.")
    cached = set(map(str, cache_events[0]["nodes"]))
    if sampler in cached:
        raise ValueError(f"Sampler {sampler} was cached; cannot use this timing.")
    if mode == "leap" and not warmup and CALIBRATION not in cached:
        raise ValueError("LEAP calibration reran during a measured trial. Enable the normal ComfyUI cache "
                         "and ensure the calibration can stay cached between jobs.")
    outputs = item.get("outputs", {})
    other_outputs = sorted({arm[2] for arm in ARMS if arm[2] != terminal}.intersection(outputs))
    if other_outputs:
        raise ValueError(f"ComfyUI returned unexpected SaveImage terminals {other_outputs}; "
                         "a benchmark trial must execute just the selected arm.")
    saved = outputs.get(terminal, {}).get("images")
    if not isinstance(saved, list) or not saved:
        raise ValueError(f"Missing SaveImage output from terminal {terminal}.")
    texts = outputs.get(sampler, {}).get("text")
    if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError(f"Missing JSON metrics at sampler node {sampler}.")
    report = json.loads(texts[0])
    for k, value in dict(schema_version=1, mode=mode, run_nonce=nonce, seed=seed,
                         steps=steps, guided_calls=guided).items():
        if not isinstance(report, dict) or type(report.get(k)) is not type(value) or report[k] != value:
            raise ValueError(f"Unexpected {k} in sampler report for node {sampler}.")
    forwards, seconds = report.get("physical_forwards"), report.get("sampler_seconds")
    if type(forwards) is not int or forwards < guided:
        raise ValueError("Sampler report has no valid real forward count.")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Sampler report has no valid sampler time.")
    batches = report.get("forward_batch_sizes")
    if (not isinstance(batches, list) or len(batches) != forwards or
            any(type(size) is not int or size < 1 for size in batches)):
        raise ValueError("Sampler report has no valid physical forward batch sizes.")
    sigmas = report.get("sigmas")
    if (not isinstance(sigmas, list) or len(sigmas) != steps + 1 or
            any(type(s) not in (float, int) or not math.isfinite(s) for s in sigmas) or
            sigmas[-1] != 0 or any(a <= b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError("Sampler report has an invalid Euler sigma grid.")
    return report, saved


def _one_trial(graph, request, arm, nonce, seed, timeout, warmup):
    _, sampler, terminal, _, _, _ = arm
    queue = request("GET", "/queue", timeout=min(10, timeout))
    if (queue.get("queue_running") != [] or queue.get("queue_pending") != []):
        raise ValueError("ComfyUI queue is not idle. Stop submitting other jobs during the benchmark.")
    trial_graph = copy.deepcopy(graph)
    trial_graph[sampler]["inputs"]["run_nonce"] = nonce
    start = time.perf_counter()
    receipt = request("POST", "/prompt", {"prompt": trial_graph, "partial_execution_targets": [terminal]},
                      timeout=min(10, timeout))
    prompt_id = receipt.get("prompt_id")
    if not isinstance(prompt_id, str) or str(uuid.UUID(prompt_id)) != prompt_id:
        raise ValueError("Local ComfyUI did not return a valid prompt ID.")
    while (remaining := timeout - (time.perf_counter() - start)) > 0:
        matches = request("GET", f"/history/{prompt_id}", timeout=min(10, remaining))
        if item := matches.get(prompt_id):
            wall = time.perf_counter() - start
            report, images = _fresh_report(item, prompt_id, arm, nonce, seed, warmup)
            return {"arm": arm[0], "prompt_id": prompt_id, "sampler_node": sampler,
                    "output_node": terminal, "queue_to_history_seconds": wall, "images": images,
                    "sampler": report}
        time.sleep(min(remaining, POLL_SECONDS))
    raise TimeoutError(f"Prompt {prompt_id} timed out and may still be running; check ComfyUI. No automatic retry.")


def _checkpoint(path, result):
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Use a same-directory unique temporary so readers never see a partial checkpoint.
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_benchmark(graph, *, request, repeats, timeout, output):
    seed = validate_graph(graph)
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("Repeat count must be 1–100.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Per-job timeout must be positive and finite.")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists; choose another path: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output.parent}")

    result = {"schema_version": 1, "completed": False, "seed": seed,
              "queue_to_history_note": "Starts before POST /prompt; includes queue, graph execution, "
                    "image saving, HTTP and the observation poll. Warmups excluded from summaries.",
              "warmups": [], "trials": [], "summary": {}}
    base_grid = None
    nonce = secrets.randbelow((1 << 31) - 3 * (repeats + 1) - 1) + 1
    for run in range(-1, repeats):
        order = ARMS if run < 0 else ARMS[run % 3:] + ARMS[:run % 3]
        dest = "warmups" if run < 0 else "trials"
        for arm in order:
            row = _one_trial(graph, request, arm, nonce, seed, timeout, run < 0)
            grid = row["sampler"]["sigmas"]
            if arm[0] != "euler15":
                if base_grid is None:
                    base_grid = grid
                if grid != base_grid:
                    raise ValueError("Measured Euler-25 and LEAP-25 sigma grids do not match.")
            row["round"] = run + 1
            result[dest].append(row)
            _checkpoint(output, result)
            nonce += 1
    for name, *_ in ARMS:
        rows = [r for r in result["trials"] if r["arm"] == name]
        samples = [r["sampler"]["sampler_seconds"] for r in rows]
        forwards = [r["sampler"]["physical_forwards"] for r in rows]
        result["summary"][name] = {
            "trials": len(rows), "median_sampler_seconds": statistics.median(samples),
            "sampler_seconds_min": min(samples), "sampler_seconds_max": max(samples),
            "median_queue_to_history_seconds": statistics.median(r["queue_to_history_seconds"] for r in rows),
            "physical_forwards_min": min(forwards), "physical_forwards_max": max(forwards),
        }
    result["completed"] = True
    _checkpoint(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path, help="API-format export of the supplied moss_leap_compare workflow")
    parser.add_argument("--port", type=int, default=8188, help="local ComfyUI port (default 8188)")
    parser.add_argument("--repeats", type=int, default=3, help="measured trials per arm (default 3)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per job (default 3600)")
    parser.add_argument("--output", type=Path, default=Path("leap_benchmark.json"), help="new JSON output path")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1–65535")
    try:
        graph = json.loads(args.graph.read_text(encoding="utf-8-sig"))
        print("Use an idle local ComfyUI queue with its normal cache. Warmup images will also be saved.",
              flush=True)
        result = run_benchmark(graph, request=local_request(args.port), repeats=args.repeats,
                               timeout=args.timeout, output=args.output)
    except (OSError, ValueError, TypeError, http.client.HTTPException) as exc:
        print(f"Benchmark stopped: {exc}", file=sys.stderr)
        print("Check the local queue before retrying. Successful trials, if any, remain in the checkpoint.",
              file=sys.stderr)
        return 1
    for name, row in result["summary"].items():
        print(f"{name}: sampler {row['median_sampler_seconds']:.3f} s; "
              f"queue→history {row['median_queue_to_history_seconds']:.3f} s (medians)")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
