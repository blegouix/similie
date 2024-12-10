# About {#mainpage}
<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

# SimiLie
**Github repo [here](https://github.com/blegouix/similie).**

SimiLie is a performance-portable (CPU & GPU) C++20 library aiming to extent the capabilities of [Kokkos](https://github.com/kokkos/kokkos) and [DDC](https://github.com/CExA-project/ddc) to offer a complete toolkit able to perfom tensor calculus, differential calculus and solving PDE. It should be able to address any multiphysical problem from eventually-relativistic classical field theory (in particular: solid & fluid mechanics, electromagnetism and gravitation).

\important Absolute WIP, there is no guarantee for proper working.

## Development plan

1. Short term: simulate electromagnetism on manifold.
2. Mid term (months): simulate geometrodynamics.
3. Long term: generalize to all kinds of *classical* gauge theories.
4. Extra-long term (years): quantization.

## References

- [Gravitation](https://ui.adsabs.harvard.edu/abs/1973grav.book.....M/abstract). By C.W. Misner, K.S. Thorne and J.A. Wheeler. ISBN 0-7167-0334-3, ISBN 0-7167-0344-0 (pbk). San Francisco: W.H. Freeman and Company, 1973
- Desbrun, M., Kanso, E., Tong, Y. (2008). [Discrete Differential Forms for Computational Modeling](https://link.springer.com/chapter/10.1007/978-3-7643-8621-4_16). In: Bobenko, A.I., Sullivan, J.M., Schröder, P., Ziegler, G.M. (eds) Discrete Differential Geometry. Oberwolfach Seminars, vol 38. Birkhäuser Basel.
- [Finite element exterior calculus](https://doi.org/10.1137/1.9781611975543). Douglas N. Arnold. CBMS-NSF Regional Conference Series in Applied Mathematics. SIAM 2018.
- Celada, M., González, D., & Montesinos, M. (2016). [BF gravity](https://arxiv.org/abs/1610.02020). Classical and Quantum Gravity, 33(21), 213001.
- [Group Theory, Birdtracks, Lie’s, and Exceptional Groups](https://birdtracks.eu/), Predrag Cvitanović, Princeton University Press July 2008.
