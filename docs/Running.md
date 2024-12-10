# Running SimiLie {#Running}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

# General usage 

SimiLie relies on `git submodule` to manage its dependencies and on `cmake` as a build system. It is a lightweight header-only library. It can thus be included as a dependency of your project with the `cmake` command-line:

```
add_subdirectory(vendor/similie)
```

By including the header file `similie.hpp` in your code, you get access to:

- [Kokkos](https://github.com/kokkos/kokkos) in the `Kokkos::` namespace, a performance-portable (CPU & GPU) C++ library targeting all major HPC platforms.
- [DDC](https://github.com/CExA-project/ddc) in the `ddc::` namespace, an abstraction layer build upon Kokkos to provide labeling of dimensions of arrays and convenient API to describe discrete scalar fields (+ computation kernels for FFT and splines).
- [SimiLie](https://github.com/blegouix/similie) itself in the `sil::` namespace.

Moreover, parts of SimiLie relie on the FFT and splines kernels provided by DDC. The first is build upon [kokkos-fft](https://github.com/kokkos/kokkos-fft) and the second upon [kokkos-kernels](https://github.com/kokkos/kokkos-kernels) or [Ginkgo](https://github.com/ginkgo-project/ginkgo), depending on the linear algebra backend you want to use. Unfortunalty, at this moment DDC does not support linking only one of those two backends, thus you need a complete install of Ginkgo 1.8 to run SimiLie even if you do not plan to use it (this dependency is not provided as a `git submodule`, due to its weight). 

Finally, the `sil::young_tableau` module relies on the [embed](https://en.cppreference.com/w/c/preprocessor/embed) directive from C23. As it is not yet in C++ (even in C++26), the only supported compiler for this module at the moment seems to be [Clang 19](https://github.com/llvm/llvm-project/releases). Thus, by default the `BUILD_YOUNG_TABLEAU` flag is turned `OFF` and this feature (which is quite independant of the rest of the SimiLie library) is not compiled, making SimiLie a C++20 library.

\important Kokkos backends other than Serial are not yet supported.

# Compilation 

First, be sure to have a full installation of Ginkgo. Then clone the main branch of the [github repo](https://github.com/blegouix/similie) and populate the other dependencies with:

```
git clone --recurse-submodules git@github.com:blegouix/similie.git
```

Make a `build/` folder and run the commands:

```
cd build/
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug ..
```

\important Debug mode is strongly recommended during development because a lot of assertions guarantee the correct usage of library. Running in Release mode without having checked first the proper execution in Debug mode may result to silent bugs. Release mode is of course required at a later stage to get full performance.

You can then edit the configuration with `ccmake .`. Please refer to the documentations of Kokkos and DDC for their respective cmake compilation flags.

Finally compile with the `make` command (with `-j` flag for parallel compilation).

# Usage

Tests chain can be run with:

```
ctest
```

A particular test with:

```
ctest -R NameOfTheTest --verbose
```

And examples are compiled in the `examples/` folder.
