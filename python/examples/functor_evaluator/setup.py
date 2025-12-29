# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

from pathlib import Path

from setuptools import Extension, setup
from Cython.Build import cythonize

root = Path(__file__).resolve().parents[2]

extensions = [
    Extension(
        name="cython_functor",
        sources=["cython_functor.pyx"],
        language="c++",
        include_dirs=[str(root / "include")],
    ),
]

setup(
    name="cython-functor-example",
    ext_modules=cythonize(extensions, language_level="3"),
)
