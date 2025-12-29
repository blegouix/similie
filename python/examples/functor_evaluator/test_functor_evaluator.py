# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def build_extension():
    here = Path(__file__).parent
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=here,
        check=True,
    )


def test_cython_functor_via_cpp():
    build_extension()
    cython_functor = importlib.import_module("cython_functor")

    handle = cython_functor.cython_functor_create(0.5)
    try:
        value = cython_functor.evaluate_from_cpp(handle, 52, 34, 1)
    finally:
        cython_functor.cython_functor_destroy(handle)

    assert value == pytest.approx(246.0)
