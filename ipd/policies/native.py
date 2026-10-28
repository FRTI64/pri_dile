"""Optional exact-order native CPU inference for deployable policies.

The C++ translation in ``policy_ops.cpp`` reproduces the Python reference's
left-to-right double additions and libm ``tanh``/``exp`` calls.  It is built
with FMA contraction, reassociation and vectorisation disabled, so using it
must not change a single score, action or probability.

If NumPy or a C++ compiler is unavailable, the helpers return ``None`` and the
policy classes keep their original Python path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ipd.paths import ROOT

try:  # optional, mirroring the fast-rule native module
    import numpy as np
except ImportError:  # pragma: no cover - minimal installations
    np = None

_SOURCE = Path(__file__).with_name("policy_ops.cpp")
_LIB: ctypes.CDLL | None = None
_LIB_ERROR: str | None = None


def _disabled() -> bool:
    return os.environ.get("IPD_FAST_POLICY", "").lower() in ("0", "off",
                                                             "false")


def _build_root() -> Path:
    override = os.environ.get("IPD_FAST_POLICY_BUILD_DIR")
    if override:
        return Path(override)
    preferred = ROOT / ".build" / "fast_policy"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ipd_fast_policy"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _compiler() -> str | None:
    for name in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if name and shutil.which(name):
            return name
    return None


def _build_library() -> Path:
    source = _SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    target = _build_root() / f"policy_{digest}.so"
    if target.exists():
        return target
    compiler = _compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler found (set CXX or install g++)")
    tmp = target.with_suffix(".so.tmp")
    command = [
        compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
        "-ffp-contract=off", "-fno-fast-math", "-fno-tree-vectorize",
        "-o", str(tmp), str(_SOURCE),
    ]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to build {_SOURCE.name} with {compiler}:\n"
            f"{result.stderr.strip()}")
    os.replace(tmp, target)
    return target


def load() -> ctypes.CDLL | None:
    global _LIB, _LIB_ERROR
    if _disabled() or np is None:
        return None
    if _LIB is not None:
        return _LIB
    if _LIB_ERROR is not None:
        return None
    try:
        lib = ctypes.CDLL(str(_build_library()))
        lib.pdi_mlp_forward.restype = ctypes.c_int
        lib.pdi_mlp_forward.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.pdi_gru_step.restype = ctypes.c_int
        lib.pdi_gru_step.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.pdi_selfplay_gru_step.restype = ctypes.c_int
        lib.pdi_selfplay_gru_step.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        _LIB = lib
        return _LIB
    except Exception as exc:  # pragma: no cover - depends on toolchain
        _LIB_ERROR = str(exc)
        return None


def available() -> bool:
    return load() is not None


def build_error() -> str | None:
    if _LIB is None:
        load()
    return _LIB_ERROR


def _double_array(values: Sequence[float]):
    return np.asarray(values, dtype=np.float64)


def _dims(layers: Sequence[tuple[Sequence[Sequence[float]],
                                 Sequence[float]]]):
    out = []
    for w, _b in layers:
        out.extend((len(w), len(w[0])))
    return np.asarray(out, dtype=np.int32)


def _flat(layers: Sequence[tuple[Sequence[Sequence[float]],
                                 Sequence[float]]]):
    data: list[float] = []
    for w, b in layers:
        for row in w:
            data.extend(float(v) for v in row)
        data.extend(float(v) for v in b)
    return _double_array(data)


class _CopyFallback:
    """Force copies to use the Python path instead of ctypes pointers."""

    def __deepcopy__(self, memo):
        return None

    def __copy__(self):
        return None


class NativeMLP(_CopyFallback):
    """Exact-order MLP forward for NeuralPolicy/RLPolicy weight layouts."""

    def __init__(self, layers: Sequence[tuple[Sequence[Sequence[float]],
                                              Sequence[float]]]):
        self.lib = load()
        if self.lib is None:
            raise RuntimeError("native policy inference unavailable")
        self.layers = int(len(layers))
        self._dims = _dims(layers)
        self._w = _flat(layers)
        self._x = np.empty(int(self._dims[1]), dtype=np.float64)
        self._out = np.empty(1, dtype=np.float64)
        self._dims_p = self._dims.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int))
        self._w_p = self._w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._x_p = self._x.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._out_p = self._out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))

    def forward(self, features: Sequence[float]) -> float:
        self._x[:] = features
        code = self.lib.pdi_mlp_forward(
            self._x_p, self._w_p, self._dims_p, self.layers, self._out_p)
        if code != 0:  # pragma: no cover - argument validation
            raise ValueError(f"native MLP failed with code {code}")
        return float(self._out[0])


def _gru_parts(layer: Any):
    if isinstance(layer, dict):
        return (layer["weight_ih"], layer["bias_ih"],
                layer["weight_hh"], layer["bias_hh"])
    return layer[0], layer[1], layer[2], layer[3]


def _gru_weights(gru: Sequence[Any]):
    data: list[float] = []
    for layer in gru:
        w_ih, b_ih, w_hh, b_hh = _gru_parts(layer)
        for row in w_ih:
            data.extend(float(v) for v in row)
        data.extend(float(v) for v in b_ih)
        for row in w_hh:
            data.extend(float(v) for v in row)
        data.extend(float(v) for v in b_hh)
    return _double_array(data)


class NativeGRU(_CopyFallback):
    """Exact-order GRU step plus head for GRUPolicy."""

    def __init__(self, gru: Sequence[Any], head: Sequence[Any],
                 hidden: int):
        self.lib = load()
        if self.lib is None:
            raise RuntimeError("native policy inference unavailable")
        self.hidden = int(hidden)
        self.n_layers = int(len(gru))
        self._w = _gru_weights(gru)
        self._head_dims = _dims(head)
        self._head_w = _flat(head)
        self._h = np.zeros((self.n_layers, self.hidden), dtype=np.float64)
        self.n_features = int(len(_gru_parts(gru[0])[0][0]))
        self._x = np.empty(self.n_features, dtype=np.float64)
        self._out = np.empty(1, dtype=np.float64)
        self._w_p = self._w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._head_w_p = self._head_w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._head_dims_p = self._head_dims.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int))
        self._h_p = self._h.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._x_p = self._x.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._out_p = self._out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))

    def reset(self) -> None:
        self._h.fill(0.0)

    def step(self, features: Sequence[float]) -> tuple[float, list[list[float]]]:
        self._x[:] = features
        code = self.lib.pdi_gru_step(
            self._x_p, int(self._x.shape[0]), self._h_p,
            self.n_layers, self.hidden, self._w_p,
            self._head_w_p, self._head_dims_p,
            int(self._head_dims.shape[0] // 2), self._out_p)
        if code != 0:  # pragma: no cover - argument validation
            raise ValueError(f"native GRU failed with code {code}")
        return float(self._out[0]), self._h.tolist()


class NativeSelfPlayGRU(_CopyFallback):
    """Exact-order two-head GRU for SelfPlayGRUPolicy."""

    def __init__(self, gru: Sequence[Any], actor: Sequence[Any],
                 critic: Sequence[Any], hidden: int):
        self.lib = load()
        if self.lib is None:
            raise RuntimeError("native policy inference unavailable")
        self.hidden = int(hidden)
        self.n_layers = int(len(gru))
        self._w = _gru_weights(gru)
        self._actor_dims = _dims(actor)
        self._actor_w = _flat(actor)
        self._critic_dims = _dims(critic)
        self._critic_w = _flat(critic)
        self._h = np.zeros((self.n_layers, self.hidden), dtype=np.float64)
        self.n_features = int(len(_gru_parts(gru[0])[0][0]))
        self._x = np.empty(self.n_features, dtype=np.float64)
        self._out = np.empty(2, dtype=np.float64)
        self._w_p = self._w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._actor_w_p = self._actor_w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._actor_dims_p = self._actor_dims.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int))
        self._critic_w_p = self._critic_w.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._critic_dims_p = self._critic_dims.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int))
        self._h_p = self._h.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._x_p = self._x.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))
        self._out_p = self._out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))

    def reset(self) -> None:
        self._h.fill(0.0)

    def step(self, features: Sequence[float]
             ) -> tuple[float, float, list[list[float]]]:
        self._x[:] = features
        code = self.lib.pdi_selfplay_gru_step(
            self._x_p, int(self._x.shape[0]), self._h_p,
            self.n_layers, self.hidden, self._w_p,
            self._actor_w_p, self._actor_dims_p,
            int(self._actor_dims.shape[0] // 2),
            self._critic_w_p, self._critic_dims_p,
            int(self._critic_dims.shape[0] // 2), self._out_p)
        if code != 0:  # pragma: no cover - argument validation
            raise ValueError(f"native self-play GRU failed with code {code}")
        return (float(self._out[0]), float(self._out[1]),
                self._h.tolist())


def make_mlp(layers: Sequence[Any]) -> NativeMLP | None:
    if not available():
        return None
    try:
        return NativeMLP(layers)
    except Exception:  # pragma: no cover - defensive fallback
        return None


def make_gru(gru: Sequence[Any], head: Sequence[Any],
             hidden: int) -> NativeGRU | None:
    if not available():
        return None
    try:
        return NativeGRU(gru, head, hidden)
    except Exception:  # pragma: no cover - defensive fallback
        return None


def make_selfplay_gru(gru: Sequence[Any], actor: Sequence[Any],
                      critic: Sequence[Any], hidden: int
                      ) -> NativeSelfPlayGRU | None:
    if not available():
        return None
    try:
        return NativeSelfPlayGRU(gru, actor, critic, hidden)
    except Exception:  # pragma: no cover - defensive fallback
        return None


__all__ = ["NativeMLP", "NativeGRU", "NativeSelfPlayGRU", "available",
           "build_error", "load", "make_mlp", "make_gru",
           "make_selfplay_gru"]
