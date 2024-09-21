# SimiLie
Simulation in Lie groups

This is a C++ performance-portable project, supporting CPU & GPU. For simplicity sake and performance purpose (avoid access to adjacency matrix, efficient parallelization & memory access), only mesh structured as a cartesian product of 1D meshes is supported. No adaptative mesh refinement. Main focus are modularity, versatility and performance.

**Absolute WIP, there is no guarantee for proper working**

## Development plan

1. Short term (month): simulate electromagnetism on manifold.
2. Mid term (months): simulate geometrodynamics.
3. Long term: generalize to all kinds of *classical* gauge theories and put Lie groups at the center of the construction.
4. Extra-long term (years): quantization.

## Initial roadmap

DONE:
- Create repository.

TODO:
- Initial setup (dependencies, basic test, project structure).
- Simple mesher.
- Class to describe a tensor (& tensor field ?).
- Differential forms & fields of differential forms.
- Basic useful differential forms (not involving derivatives): tetrads, metric.
- More advanced tools: discrete Hodge star, exterior derivative, codifferential operator, covariant derivative.
- More advanced useful differential forms: connections, Riemann, Ricci.
- Integration, inner product.
- Weak formulation, finite element method applied to electrostatic problem on manifold.
- Generalization to electromagnetism.
- ...
