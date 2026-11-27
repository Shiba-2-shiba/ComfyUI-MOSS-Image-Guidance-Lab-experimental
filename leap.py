"""LEAP's local Euler derivative predictor (Zheng et al., arXiv:2610.25918).

No ComfyUI imports. References use the full Euler grid; calibration aggregates
the unnormalized summed ridge objective (paper eq. 28) in CPU float64.
"""

import json
import math
from collections import deque
from types import MappingProxyType

import torch


_FORMAT = "moss-leap"
_VERSION = 1

# Operational guard for this experimental integration, not a LEAP-paper
# constant. Refuse numerically extreme fits rather than silently clamping.
_MAX_HISTORY_L1_GAIN = 64.0


def _int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"LEAP {name} must be an integer >= {minimum}.")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"LEAP {name} must be a finite number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"LEAP {name} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"LEAP {name} must be a finite number.")
    return result


def prediction_steps(steps, window=5):
    """Indices of network-free steps; always leave the final update exact."""
    _int(steps, "update count")
    _int(window, "window")
    return tuple(range(window, steps - 1, 2))


def _sigmas(values):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("LEAP sigmas must be a one-dimensional floating-point grid.")
        values = values.detach().to(device="cpu", dtype=torch.float64).tolist()
    if not isinstance(values, (tuple, list)):
        raise ValueError("LEAP sigmas must be a one-dimensional floating-point grid.")
    sigma = tuple(_finite(value, "sigma") for value in values)
    if len(sigma) < 2 or sigma[-1] != 0.0 or any(a <= b for a, b in zip(sigma, sigma[1:])):
        raise ValueError("LEAP sigmas must strictly decrease and end in zero.")
    return sigma


def _metadata_json(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("LEAP metadata must be a JSON object.")
    try:
        return json.dumps(metadata, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("LEAP metadata must be finite and JSON compatible.") from error


class _Configuration:
    def __init__(self, sigmas, latent_shape, metadata, window, ridge):
        self._sigmas = _sigmas(sigmas)
        if not isinstance(latent_shape, (tuple, list)) or not latent_shape:
            raise ValueError("LEAP latent shape must contain positive integers.")
        self._latent_shape = tuple(_int(s, "latent shape dimension") for s in latent_shape)
        self._metadata_json = _metadata_json(metadata)
        self._window = _int(window, "window")
        self._ridge = _finite(ridge, "ridge coefficient")
        if self._ridge <= 0:
            raise ValueError("LEAP ridge coefficient must be positive.")
        self._predictions = prediction_steps(len(self.sigmas) - 1, self.window)
        if not self._predictions:
            raise ValueError("LEAP grid must allow at least one prediction and a final exact update.")

    @property
    def sigmas(self):
        return self._sigmas

    @property
    def latent_shape(self):
        return self._latent_shape

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def window(self):
        return self._window

    @property
    def ridge(self):
        return self._ridge

    def _check_state(self, sigmas, latent_shape):
        if _sigmas(sigmas) != self.sigmas:
            raise ValueError("LEAP calibration sigma grid does not match the sampler sigma grid.")
        if tuple(latent_shape) != self.latent_shape:
            raise ValueError("LEAP calibration latent shape does not match.")

    def validate(self, sigmas, latent_shape, metadata):
        self._check_state(sigmas, latent_shape)
        if _metadata_json(metadata) != self._metadata_json:
            raise ValueError("LEAP calibration metadata does not match.")

    def _header(self, kind, references):
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "kind": kind,
            "sigmas": list(self.sigmas),
            "latent_shape": list(self.latent_shape),
            "metadata": self.metadata,
            "window": self.window,
            "ridge": self.ridge,
            "reference_count": references,
        }

    @staticmethod
    def _read_header(record, kind, minimum_references):
        if (
            not isinstance(record, dict)
            or record.get("format") != _FORMAT
            or type(record.get("version")) is not int
            or record["version"] != _VERSION
            or record.get("kind") != kind
        ):
            raise ValueError("Invalid or unsupported LEAP record format.")
        try:
            refs = _int(record["reference_count"], "reference count", minimum_references)
            return refs, {k: record[k] for k in ("sigmas", "latent_shape", "metadata", "window", "ridge")}
        except KeyError as error:
            raise ValueError(f"Missing LEAP record field: {error.args[0]}.") from error


class LeapBank(_Configuration):
    def __init__(self, sigmas, *, latent_shape, metadata, window, ridge, reference_count, coefficients):
        super().__init__(sigmas, latent_shape, metadata, window, ridge)
        self._reference_count = _int(reference_count, "reference count")
        if not isinstance(coefficients, dict) or set(coefficients) != set(self._predictions):
            raise ValueError("LEAP coefficient steps do not match the predict/correct schedule.")
        rows = {}
        for index in self._predictions:
            row = coefficients[index]
            if not isinstance(row, (tuple, list)) or len(row) != self.window:
                raise ValueError(f"LEAP coefficient row {index} has the wrong window size.")
            row = tuple(_finite(c, f"coefficient at step {index}") for c in row)
            # Bound both the absolute weights applied during the difference-
            # basis arithmetic and the equivalent projection on raw history.
            # The first also rejects huge near-cancelling coefficients before
            # the raw-basis coefficients are formed.
            if any(abs(c) > _MAX_HISTORY_L1_GAIN for c in row) or sum(abs(c) for c in row) > _MAX_HISTORY_L1_GAIN:
                raise ValueError(f"LEAP coefficient row {index} exceeds the numerical safety gain limit of 64.")
            raw_weights = (row[0],) + tuple(b - a for a, b in zip(row, row[1:]))
            if sum(abs(c) for c in raw_weights) > _MAX_HISTORY_L1_GAIN:
                raise ValueError(f"LEAP coefficient row {index} exceeds the numerical safety gain limit of 64.")
            rows[index] = row
        self._coefficients = MappingProxyType(rows)

    @property
    def reference_count(self):
        return self._reference_count

    @property
    def coefficients(self):
        return self._coefficients

    def to_dict(self):
        return self._header("coefficients", self.reference_count) | {
            "coefficients": {str(i): list(c) for i, c in self.coefficients.items()},
        }

    @classmethod
    def from_dict(cls, record):
        refs, config = cls._read_header(record, "coefficients", 1)
        rows = record.get("coefficients")
        if not isinstance(rows, dict) or any(type(k) is not str or not k.isascii() or not k.isdigit() for k in rows):
            raise ValueError("Invalid LEAP coefficient rows.")
        keys = [str(i) for i in prediction_steps(len(_sigmas(config["sigmas"])) - 1, config["window"])]
        if set(rows) != set(keys):
            raise ValueError("LEAP coefficient steps do not match the predict/correct schedule.")
        return cls(reference_count=refs, coefficients={int(k): rows[k] for k in keys}, **config)


def _symmetric_gram(gram):
    upper = gram.transpose(-2, -1)
    if torch.equal(gram, upper):
        return gram
    # Halve first so two valid large entries do not overflow when added.
    return gram * 0.5 + upper * 0.5


def _basis(history):
    """Equation (26), oldest-minus-next differences then the most recent value."""
    values = tuple(history)
    return [a - b for a, b in zip(values, values[1:])] + [values[-1]]


class LeapAccumulator(_Configuration):
    """Sufficient statistics; does not retain full trajectories or input tensors."""

    def __init__(self, sigmas, *, latent_shape, metadata, window=5, ridge=1e-4):
        super().__init__(sigmas, latent_shape, metadata, window, ridge)
        self._reference_count = 0
        n = len(self._predictions)
        self._gram = torch.zeros((n, window, window), dtype=torch.float64, device="cpu")
        self._rhs = torch.zeros((n, window), dtype=torch.float64, device="cpu")

    @property
    def reference_count(self):
        return self._reference_count

    def add_trajectory(self, velocities):
        """Consume all true Euler derivatives in chronological order, atomically."""
        hist = deque(maxlen=self.window)
        gram, rhs = torch.zeros_like(self._gram), torch.zeros_like(self._rhs)
        fit_row = {index: row for row, index in enumerate(self._predictions)}
        count = 0
        for i, value in enumerate(velocities):
            count += 1
            if count >= len(self.sigmas):
                raise ValueError("LEAP calibration trajectory is longer than the sigma grid.")
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.shape != self.latent_shape
            ):
                raise ValueError(f"LEAP calibration velocity shape or dtype is invalid at step {i}.")
            flat = value.detach().to(device="cpu", dtype=torch.float64, copy=True).flatten()
            if not torch.isfinite(flat).all():
                raise ValueError(f"LEAP calibration velocity is non-finite at step {i}.")
            row = fit_row.get(i)
            if row is not None:
                B = torch.stack(_basis(hist), dim=-1)
                gram[row] = B.T @ B
                rhs[row] = B.T @ flat
            hist.append(flat)
        if count != len(self.sigmas) - 1:
            raise ValueError("LEAP calibration trajectory is shorter than the sigma grid.")
        self._accumulate(gram, rhs, 1)

    def _accumulate(self, gram, rhs, references):
        summed_gram, summed_rhs = self._gram + _symmetric_gram(gram), self._rhs + rhs
        if not torch.isfinite(summed_gram).all() or not torch.isfinite(summed_rhs).all():
            raise ValueError("LEAP sufficient statistics overflowed.")
        self._gram, self._rhs = summed_gram, summed_rhs
        self._reference_count += references

    def merge(self, other):
        if not isinstance(other, LeapAccumulator) or other is self:
            raise ValueError("LEAP requires a separate compatible accumulator to merge.")
        self.validate(other.sigmas, other.latent_shape, other.metadata)
        if self.window != other.window or self.ridge != other.ridge:
            raise ValueError("LEAP accumulator window or ridge coefficient does not match.")
        self._accumulate(other._gram, other._rhs, other.reference_count)

    def finish(self):
        if not self.reference_count:
            raise ValueError("LEAP calibration requires at least one reference trajectory.")
        system = self._gram + self.ridge * torch.eye(self.window, dtype=torch.float64, device="cpu")
        try:
            coefficients = torch.linalg.solve(system, self._rhs.unsqueeze(-1)).squeeze(-1)
        except RuntimeError as error:
            raise ValueError("LEAP ridge fit failed: the calibration system is singular or invalid.") from error
        return LeapBank(
            self.sigmas, latent_shape=self.latent_shape, metadata=self.metadata,
            window=self.window, ridge=self.ridge, reference_count=self.reference_count,
            coefficients=dict(zip(self._predictions, coefficients.tolist())),
        )

    def to_dict(self):
        return self._header("statistics", self.reference_count) | {
            "gram": self._gram.tolist(), "rhs": self._rhs.tolist(),
        }

    @classmethod
    def from_dict(cls, record):
        refs, config = cls._read_header(record, "statistics", 0)
        result = cls(**config)
        try:
            gram = torch.tensor(record["gram"], dtype=torch.float64, device="cpu")
            rhs = torch.tensor(record["rhs"], dtype=torch.float64, device="cpu")
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError("Invalid LEAP statistics.") from error
        if (
            gram.shape != result._gram.shape or rhs.shape != result._rhs.shape
            or not torch.isfinite(gram).all() or not torch.isfinite(rhs).all()
            or (refs == 0 and (torch.count_nonzero(gram) or torch.count_nonzero(rhs)))
        ):
            raise ValueError("Invalid LEAP statistics shape, values, symmetry, or reference count.")
        # BLAS Gram reductions can round mirrored results in a different order.
        # Accept small differences, then store the same symmetric matrix we fit.
        scale = gram.abs().amax(dim=(-2, -1), keepdim=True)
        scale = torch.where(scale == 0, 1.0, scale)
        scaled_gram = gram / scale
        tolerance = 64 * torch.finfo(gram.dtype).eps * result.window
        if ((scaled_gram - scaled_gram.transpose(-2, -1)).abs() > tolerance).any():
            raise ValueError("Invalid LEAP statistics: Gram matrix is not symmetric.")
        gram = _symmetric_gram(gram)
        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        if (diagonal < 0).any() or (torch.linalg.eigvalsh(gram / scale) < -tolerance).any():
            raise ValueError("Invalid LEAP statistics: Gram matrix is not positive semidefinite.")
        if ((diagonal == 0) & (rhs != 0)).any():
            raise ValueError("Invalid LEAP statistics: nonzero RHS for a zero Gram column.")
        result._accumulate(gram, rhs, refs)
        return result


def _euler(model, x, sigmas, extra_args, callback, bank, on_velocity):
    s_in = x.new_ones([x.shape[0]])
    kwargs = {} if extra_args is None else extra_args
    history = deque(maxlen=bank.window if bank is not None else 0)
    coefficients = bank.coefficients if bank is not None else {}
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        row = coefficients.get(i)
        if row is None:
            denoised = model(x, sigma * s_in, **kwargs)
            d = (x - denoised) / sigma
        else:
            # A native correction uses the current predicted path, not a stale
            # reference state. Predicted derivatives enter history just like true ones.
            basis = _basis(history)
            pred = basis[-1] * row[-1]
            for column, coefficient in zip(basis[:-1], row[:-1]):
                pred.add_(column, alpha=coefficient)
            d = pred.to(dtype=x.dtype)
            denoised = x - d * sigma if callback is not None else None
        if bank is not None:
            history.append(d.to(dtype=work_dtype))
        if on_velocity is not None:
            on_velocity(i, d)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        x = x + d * (sigmas[i + 1] - sigma)
    return x


@torch.no_grad()
def sample_euler_reference(model, x, sigmas, extra_args=None, callback=None, disable=None, *, on_velocity=None):
    """Full deterministic Euler trajectory for calibration or a control arm."""
    _sigmas(sigmas)
    return _euler(model, x, sigmas, extra_args, callback, None, on_velocity)


@torch.no_grad()
def sample_leap_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, *, bank):
    """LEAP on the full Euler grid using a validated, condition-matched bank."""
    if not isinstance(bank, LeapBank):
        raise ValueError("LEAP requires a fitted coefficient bank.")
    bank._check_state(sigmas, x.shape)
    return _euler(model, x, sigmas, extra_args, callback, bank, None)
