<!--
SPDX-FileCopyrightText: 2024 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0
-->

# SimiLie
Simulation in Lie groups.

Documentation [here](https://blegouix.github.io/SimiLie/).

This is a C++ performance-portable project, supporting CPU & GPU. For simplicity sake and performance purpose (avoid access to adjacency matrix, efficient parallelization & memory access), only mesh structured as a cartesian product of 1D meshes is supported. No adaptative mesh refinement. Main focuses are modularity, versatility and performance.

Relies on [DDC](https://github.com/CExA-project/ddc).

**Absolute WIP, there is no guarantee for proper working.**

**Relies on the #embed directive from C23. As it is not yet in C++ (even in C++26), the only supported compiler seems to be Clang 19.**

## Development plan

1. Short term (month): simulate electromagnetism on manifold.
2. Mid term (months): simulate geometrodynamics.
3. Long term: generalize to all kinds of *classical* gauge theories and put Lie groups at the center of the construction.
4. Extra-long term (years): quantization.

## Initial roadmap

DONE:
- Create repository.
- Initial setup (dependencies, basic test, project structure).
- Simple mesher.
- Tensors.

TODO:
- Differential forms & fields of differential forms.
- Basic useful differential forms (not involving derivatives): tetrads, metric.
- More advanced tools: discrete Hodge star, exterior derivative, codifferential operator, covariant derivative.
- More advanced useful differential forms: connections, Riemann, Ricci.
- Integration, inner product.
- Weak formulation, finite element method applied to electrostatic problem on manifold.
- Generalization to electromagnetism.
- ...

## References

- [Gravitation](https://ui.adsabs.harvard.edu/abs/1973grav.book.....M/abstract). By C.W. Misner, K.S. Thorne and J.A. Wheeler. ISBN 0-7167-0334-3, ISBN 0-7167-0344-0 (pbk). San Francisco: W.H. Freeman and Company, 1973
- Desbrun, M., Kanso, E., Tong, Y. (2008). [Discrete Differential Forms for Computational Modeling](https://link.springer.com/chapter/10.1007/978-3-7643-8621-4_16). In: Bobenko, A.I., Sullivan, J.M., Schröder, P., Ziegler, G.M. (eds) Discrete Differential Geometry. Oberwolfach Seminars, vol 38. Birkhäuser Basel.
- [Finite element exterior calculus](https://doi.org/10.1137/1.9781611975543). Douglas N. Arnold. CBMS-NSF Regional Conference Series in Applied Mathematics. SIAM 2018.
- Celada, M., González, D., & Montesinos, M. (2016). [BF gravity](https://arxiv.org/abs/1610.02020). Classical and Quantum Gravity, 33(21), 213001.
- [Group Theory, Birdtracks, Lie’s, and Exceptional Groups](https://birdtracks.eu/), Predrag Cvitanović, Princeton University Press July 2008.
