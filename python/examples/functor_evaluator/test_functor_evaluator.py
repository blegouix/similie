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

    value = cython_functor.evaluate_functor(0.5, 52, 34, 1)

    assert value == pytest.approx(246.0)
