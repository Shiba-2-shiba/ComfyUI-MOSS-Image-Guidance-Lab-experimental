"""Read a completed SCG trace without importing ComfyUI."""

import json
from pathlib import Path


MATCH_KEYS = ("seed", "shape", "sigmas", "cfg", "noise_scale", "shift", "dtype", "model",
              "model_file", "patched_weights", "weight_wrappers", "block_tune", "text_conditions",
              "fft_convention", "initial_sha256_fp32")
SCG_KEYS = ("strength", "sigma_min", "sigma_max", "reference_kind", "reference_power")


def read_run(path):
    path = Path(path)
    if path.is_dir():
        path /= "trace.jsonl"
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream, "{}"))
        records = [json.loads(line) for line in stream]
    steps = records[:-1]
    if header.get("schema_version") != 1 or not records or records[-1].get("status") != "complete":
        raise ValueError(f"Incomplete or unsupported trace: {path}")
    if (not steps or len(steps) != len(header["sigmas"]) - 1 or records[-1]["steps_written"] != len(steps)
            or [s["step"] for s in steps] != list(range(1, len(steps) + 1))):
        raise ValueError(f"Inconsistent step sequence: {path}")
    return path.parent, header, steps


def require_match(base, candidate, keys):
    for key in keys:
        if base[key] != candidate[key]:
            raise ValueError(f"Runs differ in {key}. Use matching control and SCG inputs.")


def require_scg_match(base, candidate):
    spaces = {r.get("spectrum_space", "packed") for r in (base, candidate)}
    if spaces == {"packed_measured", "spatial_measured"}:
        require_match(base, candidate, ("strength", "sigma_min", "sigma_max", "reference_kind"))
        if not base.get("paired_reference_id") or base["paired_reference_id"] != candidate.get("paired_reference_id"):
            raise ValueError("Packed and spatial controls must use the same paired reference images.")
    elif len(spaces) == 1:
        require_match(base, candidate, SCG_KEYS)
    else:
        raise ValueError("Cross-space comparison needs the measured packed and spatial reference pair.")
