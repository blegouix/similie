<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

# SimiLie
Documentation [here](https://blegouix.github.io/similie/).

SimiLie (aka. Simulation in Lie groups) is a performance-portable (CPU & GPU) C++ library aiming to extent the capabilities of [DDC](https://github.com/CExA-project/ddc) to offer a complete toolkit able to address any physical problem from eventually-relativistic classical field theory (in particular: solid & fluid mechanics, electromagnetism and gravitation, or coupling between those).

**Absolute WIP, there is no guarantee for proper working.**

**Relies on the [#embed](https://en.cppreference.com/w/c/preprocessor/embed) directive from C23. As it is not yet in C++ (even in C++26), the only supported compiler at the moment seems to be [Clang 19](https://github.com/llvm/llvm-project/releases). [Kokkos](https://github.com/kokkos/kokkos) backends other than `Serial` are not yet supported.**

## Development plan

1. Short term: simulate electromagnetism on manifold.
2. Mid term (months): simulate geometrodynamics.
3. Long term: generalize to all kinds of *classical* gauge theories and put Lie groups and Clifford algebra at the center of the construction.
4. Extra-long term (years): quantization.

## Initial roadmap

DONE:
- Tensors (any dimension, rank and structure), tensor fields, tensor product, covariant/contravariant character, metric.
- Differential forms: (co)simplices, (co)chains, (co)boundary operator, discrete Hodge star.

TODO:
- Finite elements solver.
- Application to an electromagnetism problem on manifold.
- ...

## Contact

Send an email to baptistelegouix@gmail.com

## References

- [Gravitation](https://ui.adsabs.harvard.edu/abs/1973grav.book.....M/abstract). By C.W. Misner, K.S. Thorne and J.A. Wheeler. ISBN 0-7167-0334-3, ISBN 0-7167-0344-0 (pbk). San Francisco: W.H. Freeman and Company, 1973
- Desbrun, M., Kanso, E., Tong, Y. (2008). [Discrete Differential Forms for Computational Modeling](https://link.springer.com/chapter/10.1007/978-3-7643-8621-4_16). In: Bobenko, A.I., Sullivan, J.M., Schröder, P., Ziegler, G.M. (eds) Discrete Differential Geometry. Oberwolfach Seminars, vol 38. Birkhäuser Basel.
- [Finite element exterior calculus](https://doi.org/10.1137/1.9781611975543). Douglas N. Arnold. CBMS-NSF Regional Conference Series in Applied Mathematics. SIAM 2018.
- Celada, M., González, D., & Montesinos, M. (2016). [BF gravity](https://arxiv.org/abs/1610.02020). Classical and Quantum Gravity, 33(21), 213001.
- [Group Theory, Birdtracks, Lie’s, and Exceptional Groups](https://birdtracks.eu/), Predrag Cvitanović, Princeton University Press July 2008.
