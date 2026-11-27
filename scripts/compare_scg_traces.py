"""Compare an Euler or standard radial SCG control with an SCG trace. Write CSV to stdout."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scg_trace_io import MATCH_KEYS, read_run, require_match, require_scg_match


def relative_l2(baseline, candidate):
    difference = (candidate - baseline).reshape(len(baseline), -1)
    scale = np.linalg.norm(baseline.reshape(len(baseline), -1), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.linalg.norm(difference, axis=1) / scale
    return [float(v) if np.isfinite(v) else None for v in values]


def compare(baseline, candidate):
    base_dir, base, base_steps = read_run(baseline)
    scg_dir, scg, scg_steps = read_run(candidate)
    if base.get("within_bin_mix", 0) != 0:
        raise ValueError("Use Euler or standard radial SCG as the first run.")
    require_match(base, scg, MATCH_KEYS)
    if base["strength"]:
        require_scg_match(base, scg)
    snapshots = base["debug_log"] == scg["debug_log"] == "jsonl+latents"
    for control, step in zip(base_steps, scg_steps):
        n = step["step"]
        deltas = {}
        if snapshots:
            name = f"step_{n:04d}.npz"
            with np.load(base_dir / name, allow_pickle=False) as a, np.load(scg_dir / name, allow_pickle=False) as b:
                base_euler = a["euler"]
                base_post = a["after"] if control["reason"] == "applied" else base_euler
                scg_euler = b["euler"]
                scg_post = b["after"] if step["reason"] == "applied" else scg_euler
                deltas["denoised_ab_relative_l2"] = relative_l2(a["denoised"], b["denoised"])
                deltas["euler_ab_relative_l2"] = relative_l2(base_euler, scg_euler)
                deltas["post_ab_relative_l2"] = relative_l2(base_post, scg_post)
                if base["strength"]:
                    left = (base_post - base_euler).reshape(len(base_euler), -1)
                    right = (scg_post - scg_euler).reshape(len(scg_euler), -1)
                    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
                    deltas["scg_delta_ab_cosine"] = [
                        float(dot / norm) if norm > 0 else None for dot, norm in zip((left * right).sum(1), norms)
                    ]
        for i, sample in enumerate(step["samples"]):
            control_l2 = control["samples"][i]["scg_delta_l2"]
            measured = sample["scg_delta_l2"]
            yield dict(step=n, sigma_next=step["sigma_next"], sample=i, reason=step["reason"],
                       control_space=base.get("spectrum_space", "packed"),
                       candidate_space=scg.get("spectrum_space", "packed"),
                       requested_total_l2=sample.get("requested_total_l2"), dose_scale=sample.get("dose_scale"),
                       within_bin_mix=scg.get("within_bin_mix", 0), control_scg_l2=control_l2,
                       scg_l2_ratio=measured / control_l2 if control_l2 and measured is not None else None,
                       batch_denoised_identical=control["denoised_sha256_fp32"] == step["denoised_sha256_fp32"],
                       batch_post_identical=control["post_sha256_fp32"] == step["post_sha256_fp32"],
                       control_euler_l2=control["samples"][i]["euler_delta_l2"],
                       euler_delta_l2=sample["euler_delta_l2"], scg_delta_l2=sample["scg_delta_l2"],
                       scg_over_euler=sample["scg_over_euler"], scg_over_state=sample["scg_over_state"],
                       scg_euler_cosine=sample["scg_euler_cosine"], **{k: v[i] for k, v in deltas.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Euler or radial SCG trace.jsonl or run directory")
    parser.add_argument("candidate", type=Path, help="SCG trace.jsonl or run directory")
    args = parser.parse_args()
    rows = compare(args.baseline, args.candidate)
    first = next(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(first), lineterminator="\n")
    writer.writeheader()
    writer.writerow(first)
    writer.writerows(rows)


if __name__ == "__main__":
    main()
