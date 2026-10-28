"""ctypes loader for the native fast-rule interpreter.

The shared library is compiled on first use from ``rules.cpp`` with the system
C++ compiler.  No build system, pybind11, or network access is required.  If no
compiler is available the module reports ``available() == False`` and callers
fall back to the original Python opponents.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ipd.paths import ROOT

from .spec import FastRuleSpec, TYPE_NAMES, encode_spec

_SOURCE = Path(__file__).with_name("rules.cpp")
_LIB: ctypes.CDLL | None = None
_LIB_ERROR: str | None = None
_BUILD_LOCK = False


def _build_root() -> Path:
    override = os.environ.get("IPD_FAST_BUILD_DIR")
    if override:
        return Path(override)
    preferred = ROOT / ".build" / "fast_rules"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ipd_fast_rules"
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
    target = _build_root() / f"rules_{digest}.so"
    if target.exists():
        return target
    compiler = _compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler found (set CXX or install g++)")
    tmp = target.with_suffix(".so.tmp")
    command = [
        compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
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
    """Build (once) and return the native library, or ``None`` on failure."""
    global _LIB, _LIB_ERROR
    if _LIB is not None:
        return _LIB
    if _LIB_ERROR is not None:
        return None
    try:
        path = _build_library()
        lib = ctypes.CDLL(str(path))
        lib.rb_type_count.restype = ctypes.c_int
        lib.rb_version.restype = ctypes.c_int
        lib.rb_create.restype = ctypes.c_void_p
        lib.rb_create.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.rb_reset_all.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.rb_act.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_int)]
        lib.rb_step.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        lib.rb_observe.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        lib.rb_debug.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int)]
        lib.rb_destroy.argtypes = [ctypes.c_void_p]
        if lib.rb_type_count() != len(TYPE_NAMES):
            raise RuntimeError(
                f"native rule type count {lib.rb_type_count()} != "
                f"Python type count {len(TYPE_NAMES)}; rebuild rules.cpp")
        _LIB = lib
        return _LIB
    except Exception as exc:  # pragma: no cover - depends on toolchain
        _LIB_ERROR = str(exc)
        return None


def available() -> bool:
    """Return whether the native interpreter can be loaded."""
    return load() is not None


def build_error() -> str | None:
    """Return the build/load error, if any (for diagnostics)."""
    if _LIB is None:
        load()
    return _LIB_ERROR


def _rng_state(obj: object) -> tuple[list[int], int]:
    rng = getattr(obj, "rng", None)
    if rng is None:
        # Deterministic rules never draw, but a zero state keeps the ABI
        # uniform.
        return [0] * 624, 0
    state = rng.getstate()
    words = list(state[1])
    if len(words) != 625:
        raise ValueError(f"unexpected RNG state length {len(words)}")
    return words[:624], int(words[624])


class NativeRuleBank:
    """A batch of fast rule opponents with per-row CPython RNG states."""

    def __init__(self, specs: Sequence[FastRuleSpec],
                 rng_objects: Sequence[object] | None = None,
                 rounds: int = 100) -> None:
        self.lib = load()
        if self.lib is None:
            raise RuntimeError(f"native rules unavailable: {build_error()}")
        self.n = len(specs)
        self.rounds = int(rounds)
        self.specs = list(specs)
        encoded = [encode_spec(spec) for spec in specs]
        n = self.n
        self._type_ids = np.asarray([row["type_id"] for row in encoded],
                                    dtype=np.int32)
        self._ip = np.asarray([row["ip"] for row in encoded],
                              dtype=np.int32).reshape(n, 16)
        self._dp = np.asarray([row["dp"] for row in encoded],
                              dtype=np.float64).reshape(n, 16)
        self._seq = np.asarray([row["seq"] for row in encoded],
                               dtype=np.int32).reshape(n, 8)
        self._seq_len = np.asarray([row["seq_len"] for row in encoded],
                                   dtype=np.int32)
        rng_objects = list(rng_objects or [None] * n)
        if len(rng_objects) != n:
            raise ValueError("rng_objects must match specs")
        mt = np.zeros((n, 624), dtype=np.uint32)
        mti = np.zeros(n, dtype=np.int32)
        for i, obj in enumerate(rng_objects):
            if obj is None:
                continue
            words, index = _rng_state(obj)
            mt[i] = words
            mti[i] = index
        self._mt = mt
        self._mti = mti
        self._rounds = np.full(n, max(1, min(100, int(rounds))),
                               dtype=np.int32)
        self._handle = self.lib.rb_create(
            self._type_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._ip.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._dp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._seq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._seq_len.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._rounds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._mt.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            self._mti.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )
        if not self._handle:
            raise MemoryError("rb_create returned NULL")
        self._out = np.zeros(n, dtype=np.int32)

    def act(self, t: int, blue: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return actions for round ``t``; histories are unchanged."""
        self.lib.rb_act(self._handle, self.n, int(t),
                        self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        return self._out

    def step(self, t: int, blue: Sequence[int] | np.ndarray) -> np.ndarray:
        """Compute one round and commit it to the native histories."""
        b = np.ascontiguousarray(np.asarray(blue, dtype=np.int32))
        if len(b) != self.n:
            raise ValueError("blue length does not match bank size")
        self.lib.rb_step(
            self._handle, self.n, int(t),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        return self._out

    def observe(self, t: int, blue: Sequence[int] | np.ndarray,
                opponent: Sequence[int] | np.ndarray) -> None:
        """Append the completed round's actions to every row's history."""
        b = np.ascontiguousarray(np.asarray(blue, dtype=np.int32))
        o = np.ascontiguousarray(np.asarray(opponent, dtype=np.int32))
        if len(b) != self.n or len(o) != self.n:
            raise ValueError("observation length does not match bank size")
        self.lib.rb_observe(
            self._handle, self.n, int(t),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

    def debug_state(self, index: int, t: int = 0) -> list[int]:
        out = (ctypes.c_int * 24)()
        self.lib.rb_debug(self._handle, int(index), int(t), out)
        return list(out)

    def reset(self) -> None:
        self.lib.rb_reset_all(self._handle, self.n)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.rb_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass


__all__ = ["NativeRuleBank", "available", "build_error", "load",
           "TYPE_NAMES"]
