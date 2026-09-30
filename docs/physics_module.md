# The physics module {#physics_module}
<!--
SPDX-FileCopyrightText: 2026 Baptiste Legouix
SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED
-->

\important This documentation is fully AI-generated.

The `similie::physics` module provides Hamiltonian equations, constitutive laws,
and quantities for scalar fields, magnetostatics, and elasticity. SymPy
definitions in `src/similie/physics/` generate C++ functors during the CMake
build. The elasticity and magnetostatics directories also contain their
physical quantities and constitutive laws.

## Scalar field

`scalar_field::ScalarFieldWithPowerCouplingHamiltonian` is generated from
the scalar field \f$\phi\f$ and its moment components \f$\pi_i\f$:

\f[
\mathcal H(\phi,\pi)
= \frac{1}{2}\left(-m^2\phi^2-\pi_0^2+\sum_{i=1}^{d-1}\pi_i^2\right)
  - \frac{g\phi^p}{\Gamma(p+1)}.
\f]

Here \f$d\f$ is the number of dimensions, \f$m\f$ is `mass`, \f$g\f$ is
`coupling_constant`, and \f$p\f$ is `coupling_power`. The generator uses a
negative sign for the first moment component and positive signs for the others.
There is no separate scalar-field constitutive-law class. The generated
Hamiltonian supplies \f$\partial\mathcal H/\partial\pi_0=-\pi_0\f$,
\f$\partial\mathcal H/\partial\pi_i=\pi_i\f$ for \f$i>0\f$, and
\f$\partial\mathcal H/\partial\phi=-m^2\phi-gp\phi^{p-1}/\Gamma(p+1)\f$.

## Magnetostatics

For magnetic vector potential \f$A\f$, induction \f$B\f$, current density
\f$j\f$, and permeability \f$\mu\f$,
`magnetostatics::LinearMagnetostaticsHamiltonian` implements

\f[
\mathcal H(A,B)=\frac{B\cdot B}{2\mu}-j\cdot A.
\f]

`LinearMagneticInductionToMagneticField` implements the corresponding discrete
constitutive map \f$H_i=(\star_i/\mu)B_i\f$, where \f$\star_i\f$ is the
supplied Hodge-star factor. The nonlinear model instead uses an interpolated
\f$B\f$–\f$H\f$ curve. `NonlinearMagnetostaticsHamiltonian` has
\f$\mathcal H(A,B)=W(|B|^2)-j\cdot A\f$, with \f$W\f$ obtained by integrating
the curve. `NonlinearMagneticInductionToMagneticField` applies its radial law
\f$H_i=\star_i\,h(|B|)B_i/|B|\f$ (with a zero-field limit). The curve also
provides derivatives used by the nonlinear solver.

## Elasticity

`elasticity::DisplacementToStrain::from_gradient` computes the symmetric strain
from a displacement gradient. `Strain2D::xy` stores tensorial shear strain,
\f$\varepsilon_{xy}=(\partial_y u_x+\partial_x u_y)/2\f$. The generated
`LinearElasticityHamiltonian` describes isotropic linear elasticity:

\f[
\mathcal H(u,\varepsilon)
= \mu\,\varepsilon:\varepsilon
  + \frac{\lambda}{2}(\operatorname{tr}\varepsilon)^2-f\cdot u,
\qquad
\mu=\frac{E}{2(1+\nu)},\quad
\lambda=\frac{E\nu}{(1+\nu)(1-(d-1)\nu)}.
\f]

\f$E\f$ is Young's modulus, \f$\nu\f$ is Poisson's ratio, and \f$f\f$ is
the body force. The generated 2D form uses plane stress; the 3D form uses the
three-dimensional Lamé coefficient. The generated
`LinearElasticStrainToStress` law has the component form
\f$\sigma_{ij}=2\mu\varepsilon_{ij}+\lambda\operatorname{tr}(\varepsilon)\delta_{ij}\f$;
its `stiffness` and `trace_coupling` parameters represent \f$2\mu\f$ and
\f$\lambda\f$. `CauchyStress2D` stores the resulting stress and provides
`von_mises()`.

### Local material Hodge in 2D

\important This operator and documentation is fully AI-generated.

`elasticity::ElasticMaterialHodge2D` is a local map for one oriented
quadrilateral cell. Construct it from four vertex positions in bit-mask order
`(0, 1, 2, 3)` and a callable `stress_law(Strain2D)` returning
`CauchyStress2D`. `edge_differences<SpatialX, SpatialY>(nodal)` uses the
exterior covariant derivative with identity transport to collect both
displacement components on the four edges `0→1`, `2→3`, `0→2`, and `1→3`.
`matrix()` returns the 8×8 material map from these edge differences to
integrated forces on the corresponding dual half-segments. Its affine part
reproduces constant-strain tractions, while a complementary term stabilizes
the remaining edge modes. The constructor rejects folded or degenerate cells
and a material law with nonpositive shear modulus.

`recover_gradient(differences)` returns the four affine displacement-gradient
components used for diagnostics. The [ONELAB elasticity
example](https://github.com/blegouix/similie/tree/main/examples/onelab_elasticity)
shows how the local map contributes to a nodal force balance.
