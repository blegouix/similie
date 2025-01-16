# Finite element module {#finite_element_module}
<!--
SPDX-FileCopyrightText: 2025 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

## Why partial differential equations are importants for physics ?

Given a differential form \f$u\f$ parametrized by coordinates \f$x\f$, any [partial differential equation](https://en.wikipedia.org/wiki/Partial_differential_equation) can be written:

\f\[
\mathcal{F}(D^k u, D^{k-1} u,... ,D u, u, x) = 0
\f\]

Where \f$D^k\f$ is a \f$k-\f$order differential operator. Any partial differential equation can be splitted into a Hamiltonian and non-Hamiltonian parts:

\f\[
\frac{\partial u}{\partial t} + \{\mathcal{H}, u\} = \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x) 
\f\]

Where \f$\{\mathcal{H}, u\}\f$ is the Poisson bracket between the Hamiltonian \f$\mathcal{H}(u, du, t)\f$ of the system and \f$u\f$. It can always be written \f$\{\mathcal{H}, u\} = \mathcal{J}\frac{\delta \mathcal{H}}{\delta u}\f$ in term of the symplectic matrix \f$\mathcal{J}\f$ (where \f$\frac{\delta}{\delta u}\f$ is a [variational derivative](https://en.wikipedia.org/wiki/Functional_derivative)). The Hamiltonian part of the PDE is associated to conserved quantities such as mass, momentum or energy.

\note If a PDE involves a second-order temporal derivative \f$\frac{\partial^2 u}{\partial t^2}\f$ (like in wave equation) we can always define \f$U = (\frac{\partial u}{\partial t}, u)\f$ and write the PDE in term of  \f$U\f$.

Partial differential equations are the main mathematical tool to describe rules of classical physics.

## What is Finite Element Method ?

Finite Element Method aims to discretize PDEs to produce numerical schemes able to solve them. It relies on the weak formulation of the PDEs (\f$\left<\cdot, \cdot\right>_\Omega = \int_\Omega \cdot \wedge \star \cdot\;\f$ is the inner product between forms on the domain \f$\Omega\f$):

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\delta \mathcal{H}}{\delta u}, \psi_j\right>_\Omega = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

In which the variational derivative of the Hamiltonian can be expanded as: 

\f\[
\frac{\delta \mathcal{H}}{\delta u} = \frac{\partial \mathcal{H}}{\partial u} - d\frac{\partial \mathcal{H}}{\partial (du)}
\f\]

Leading to:

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial u}, \psi_j\right>_\Omega + \int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}\frac{\partial \mathcal{H}}{\partial (du)} \wedge \mathrm{tr} \star \psi_j + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (du)}, d \psi_j\right>_\Omega = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

\note \f$\left< \mathcal{J}d\frac{\partial \mathcal{H}}{\partial (d u)}, \psi_j\right>_\Omega\f$ has been transformed into \f$\int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}\frac{\partial \mathcal{H}}{\partial (du)} \wedge \mathrm{tr} \star \psi_j + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (du)}, d\psi_j\right>_\Omega\f$ using integration by parts. 

Where \f$\psi\f$ is a basis of test functions.

\note As an example, the heat equation \f$\frac{\partial u}{\partial t} + \Delta u = s\f$ is associated to the Hamiltonian \f$\mathcal{H} = \frac{1}{2}\left<du,du\right>\f$ and the heat source term \f$\mathcal{D} = s\f$, leading to:

\note \f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right>_\Omega + \int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}du \wedge \mathrm{tr} \star \psi_j + \left< \mathcal{J}du, d \psi_j\right>_\Omega = \left<s, \psi_j \right>_\Omega
\f\]

We decompose \f$u\f$ and \f$\psi_j\f$ in a discrete basis function \f$\phi\f$. 

\f\[
u(x, t) = \sum_i \tilde{u}_i \phi_i(x, t)
\f\]

\attention \f$\phi\f$ and \f$\psi\f$ may or may not be the same.

The weak formulation is thus rewritten (Einstein summation convention applies):

\f\[
\frac{\partial \tilde{u}_i}{\partial t}\left<\phi_i, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\tilde{u}_i\phi_i)}, \psi_j\right>_\Omega + \int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\tilde{u}_id\phi_i)} \wedge \mathrm{tr} \star \psi_j\\ + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\tilde{u}_id\phi_i)}, d \psi_j\right>_\Omega = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

To continue the computation, we need to assume a quadratic form for the Hamiltonian:

\f\[
\mathcal{H} = \frac{\alpha}{2} \left<u|u\right> + \frac{\beta}{2} \left<du|du\right> = \tilde{u}_i^2\left(\frac{\alpha}{2} \left<\phi_i|\phi_i\right> + \frac{\beta}{2} \left<d\phi_i|d\phi_i\right>\right)

\f\]

\important If the Hamiltonian does not have this form, implementing a non-linear solver is required (it would rely on a quadratic approximation of \f$\mathcal{H}\f$, thus being able to solve the quadratic case is still necessary). 

\f\[
\frac{\partial \tilde{u}_i}{\partial t}\left<\phi_i, \psi_j\right>_\Omega + \tilde{u}_i\left(\alpha\left< \mathcal{J}\phi_i, \psi_j\right>_\Omega + \beta \int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}d\phi_i \wedge \mathrm{tr} \star \psi_j\\ + \beta \left< \mathcal{J}d\phi_i, d \psi_j\right>_\Omega\right) = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

Or in other words:

\f[
M\frac{\partial \tilde{u}}{\partial t} + K\tilde{u} = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f]

With:

\f\[
\left\{\begin{matrix}
M = \left<\phi_i, \psi_j\right>_\Omega\\
K = \alpha\left< \mathcal{J}\phi_i, \psi_j\right>_\Omega + \beta \int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}d\phi_i \wedge \mathrm{tr} \star \psi_j + \beta \left< \mathcal{J}d\phi_i, d \psi_j\right>_\Omega
\end{matrix}\right.
\f\]

(\f$M\f$ is called the <em>mass matrix</em> and \f$S\f$ the <em>stiffness matrix</em>)

\note As an example, the heat equation is associated to: 

\note \f\[
\left\{\begin{matrix}
M = \left<\phi_i, \psi_j\right>_\Omega\\
K = \int_{\partial\Omega} \mathrm{tr} \; \mathcal{J}d\phi_i \wedge \mathrm{tr} \star \psi_j + \left< \mathcal{J}d\phi_i, d \psi_j\right>_\Omega
\end{matrix}\right.
\f\]

## How is Finite Element Method implemented in Similie ?
