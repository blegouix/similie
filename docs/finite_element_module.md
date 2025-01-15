# Finite Element module {#finite_element_module}
<!--
SPDX-FileCopyrightText: 2025 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

## Why partial differential equations are importants for physics ?

Given a multidimensional function \f$u:x\rightarrow \; .\f$, any [partial differential equation](https://en.wikipedia.org/wiki/Partial_differential_equation) can be written:

\f\[
\mathcal{F}(D^k u, D^{k-1} u,... ,D u, u, x) = 0
\f\]

Where \f$D^k\f$ is a \f$k-\f$order differential operator. Any partial differential equation can be splitted into a Hamiltonian and non-Hamiltonian parts:

\f\[
\frac{\partial u}{\partial t} + \{\mathcal{H}, u\} = \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x) 
\f\]

Where \f$\{\mathcal{H}, u\}\f$ is the Poisson bracket between the Hamiltonian \f$\mathcal{H}\f$ of the system and \f$u\f$. It can always be written \f$\{\mathcal{H}, u\} = \mathcal{J}\frac{d\mathcal{H}(u, t)}{d x}\f$ in term of the symplectic matrix \f$\mathcal{J}\f$. The Hamiltonian part of the PDE is associated to conserved quantities such as mass, momentum or energy.

\note If a PDE involves a second-order temporal derivative \f$\frac{\partial^2 u}{\partial t^2}\f$ (like in wave equation) we can always define \f$U = (\frac{\partial u}{\partial t}, u)\f$ and write the PDE in term of  \f$U\f$.

Partial differential equations are the main mathematical tool to describe rules of classical physics.

## What is Finite Element Method ?

Finite Element Method aims to discretize PDE to produce numerical schemes to solve it. It relies on the weak formulation of the PDE (\f$\left<\cdot, \cdot\right>\f$ is the inner product between functions):

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right> + \left< \mathcal{J}\frac{d \mathcal{H}(u, t)}{dx}, \psi_j\right> = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>
\f\]

In which the total derivative of the Hamiltonian can be expanded using the [functional derivative](https://en.wikipedia.org/wiki/Functional_derivative):

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right> + \left< \mathcal{J}\left(\frac{\partial \mathcal{H}(u, t)}{\partial x} - \nabla\cdot \frac{\partial \mathcal{H}(u, t)}{\partial (\nabla x)}\right), \psi_j\right> = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>
\f\]

Leading to:

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right> + \left< \mathcal{J}\frac{\partial \mathcal{H}(u, t)}{\partial x}, \psi_j\right> - \left< \mathcal{J}\frac{\partial \mathcal{H}(u, t)}{\partial (\nabla x)}, \nabla \psi_j\right> = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>
\f\]


\note \f$\left< \mathcal{J}\nabla\cdot\frac{\partial \mathcal{H}(u, t)}{\partial (\nabla x)}, \psi_j\right>\f$ has been transformed into \f$\left< \mathcal{J}\frac{\partial \mathcal{H}(u, t)}{\partial (\nabla x)}, \nabla \psi_j\right>\f$ by part integration. 

Where \f$\psi\f$ is a basis of test functions.

We decompose \f$u\f$ and \f$\psi_j\f$ in a discrete basis function \f$\phi\f$. 

\f\[
u(x, t) = \sum_i \tilde{u}_i \phi_i(x, t)
\f\]

\attention \f$\phi\f$ and \f$\psi\f$ may or may not be the same.

The weak formulation is thus rewritten (Einstein summation convention applies):

\f\[
\frac{\partial \tilde{u}_i}{\partial t}\left<\phi_i, \psi_j\right> + \left< \mathcal{J}\frac{\partial \mathcal{H}(\tilde{u}_i\phi_i, t)}{\partial x}, \psi_j\right> - \left< \mathcal{J}\frac{\partial \mathcal{H}(\tilde{u}_i\phi_i, t)}{\partial (\nabla x)}, \nabla \psi_j\right> = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>
\f\]

\note As an example, the Poisson equation \f$\Delta\phi = \rho\f$ is associated to the Hamiltonian \f$\mathcal{H} = \frac{1}{2}|\nabla \phi|^2 - \rho\phi\f$.
