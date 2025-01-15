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

Where \f$\{\mathcal{H}, u\}\f$ is the Poisson bracket between the Hamiltonian \f$\mathcal{H}(u, \vec{\nabla}u, t)\f$ of the system and \f$u\f$. It can always be written \f$\{\mathcal{H}, u\} = \mathcal{J}\frac{\delta \mathcal{H}}{\delta u}\f$ in term of the symplectic matrix \f$\mathcal{J}\f$ (where \f$\frac{\delta}{\delta u}\f$ is a [variational derivative](https://en.wikipedia.org/wiki/Functional_derivative)). The Hamiltonian part of the PDE is associated to conserved quantities such as mass, momentum or energy.

\note If a PDE involves a second-order temporal derivative \f$\frac{\partial^2 u}{\partial t^2}\f$ (like in wave equation) we can always define \f$U = (\frac{\partial u}{\partial t}, u)\f$ and write the PDE in term of  \f$U\f$.

Partial differential equations are the main mathematical tool to describe rules of classical physics.

## What is Finite Element Method ?

Finite Element Method aims to discretize PDEs to produce numerical schemes able to solve them. It relies on the weak formulation of the PDEs (\f$\left<\cdot, \cdot\right>_\Omega\f$ is the inner product between functions on the domain \f$\Omega\f$):

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\delta \mathcal{H}}{\delta u}, \psi_j\right>_\Omega = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

In which the variational derivative of the Hamiltonian can be expanded as: 

\f\[
 \frac{\delta \mathcal{H}}{\delta u} = \frac{\partial \mathcal{H}}{\partial u} - \vec{\nabla}\cdot \frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}
\f\]

Leading to:

\f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial u}, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \psi_j\vec{n}\right>_{\partial\Omega} - \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \vec{\nabla} \psi_j\right>_\Omega = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]


\note \f$\left< \mathcal{J}\vec{\nabla}\cdot\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \psi_j\right>_\Omega\f$ has been transformed into \f$\left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \psi_j\vec{n}\right>_{\partial\Omega} - \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \vec{\nabla} \psi_j\right>_\Omega\f$ using integration by parts. 

Where \f$\psi\f$ is a basis of test functions.

\note As an example, the heat equation \f$\frac{\partial u}{\partial t} + \Delta u = s\f$ is associated to the Hamiltonian \f$\mathcal{H} = \frac{1}{2}|\vec{\nabla} u|^2\f$ and the heat source term \f$\mathcal{D} = s\f$, leading to:

\note \f\[
\left<\frac{\partial u}{\partial t}, \psi_j\right>_\Omega + \left< \mathcal{J}\vec{\nabla} u, \psi_j\vec{n}\right>_{\partial\Omega} - \left< \mathcal{J}\vec{\nabla} u, \vec{\nabla} \psi_j\right>_\Omega = \left<s, \psi_j \right>_\Omega
\f\]

We decompose \f$u\f$ and \f$\psi_j\f$ in a discrete basis function \f$\phi\f$. 

\f\[
u(x, t) = \sum_i \tilde{u}_i \phi_i(x, t)
\f\]

\attention \f$\phi\f$ and \f$\psi\f$ may or may not be the same.

The weak formulation is thus rewritten (Einstein summation convention applies):

\f\[
\frac{\partial \tilde{u}_i}{\partial t}\left<\phi_i, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial u}, \psi_j\right>_\Omega + \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \psi_j\vec{n}\right>_{\partial\Omega} - \left< \mathcal{J}\frac{\partial \mathcal{H}}{\partial (\vec{\nabla} u)}, \vec{\nabla} \psi_j\right>_\Omega = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

To continue the computation, we need to assume a quadratic form for the Hamiltonian:

\f\[
\mathcal{H} \sim \frac{\alpha}{2} u^2 + \frac{\beta}{2} ||\vec{\nabla} u||^2 + \gamma u||\vec{\nabla} u|| + \delta u + \epsilon ||\vec{\nabla} u|| + \zeta
\f\]

\attention If the Hamiltonian does not have this form, implementing a non-linear solver is required (it would rely on a quadratic approximation of \f$\mathcal{H}\f$, thus being able to solve the quadratic case is still necessary). 

\f\[
\frac{\partial \tilde{u}_i}{\partial t}\left<\phi_i, \psi_j\right>_\Omega + \tilde{u}_i\left(\left< \mathcal{J}\left(\alpha \phi_i+\gamma ||\vec{\nabla} \phi_i|| + \delta\right), \psi_j\right>_\Omega + \left< \mathcal{J}\left(\beta ||\vec{\nabla} \phi_i||+ \gamma \phi + \epsilon\right), \psi_j\vec{n}\right>_{\partial\Omega}\\ - \left< \mathcal{J}\left(\beta ||\vec{\nabla} \phi_i||+ \gamma \phi_i + \epsilon\right), \vec{\nabla} \psi_j\right>_\Omega\right) = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f\]

Or in other words:

\f[
M\frac{\partial \tilde{u}}{\partial t} + K\tilde{u} = \left< \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x), \psi_j \right>_\Omega
\f]

With:

\f\[
\left\{\begin{matrix}
M = \left<\phi_i, \psi_j\right>_\Omega\\
K = \left< \mathcal{J}\left(\alpha \phi_i+\gamma ||\vec{\nabla} \phi_i|| + \delta\right), \psi_j\right>_\Omega\\ + \left< \mathcal{J}\left(\beta ||\vec{\nabla} \phi_i||+ \gamma \phi + \epsilon\right), \psi_j\vec{n}\right>_{\partial\Omega}\\ - \left< \mathcal{J}\left(\beta ||\vec{\nabla} \phi_i||+ \gamma \phi_i + \epsilon\right), \vec{\nabla} \psi_j\right>_\Omega
\end{matrix}\right.
\f\]

(\f$M\f$ is called the <em>mass matrix</em> and \f$S\f$ the <em>stiffness matrix</em>)

\note As an example, the heat equation is associated to: 

\note \f\[
\left\{\begin{matrix}
M = \left<\phi_i, \psi_j\right>_\Omega\\
K = \left< \mathcal{J}\vec{\nabla} \phi_i, \psi_j\vec{n}\right>_{\partial\Omega}\\ - \left< \mathcal{J} |\vec{\nabla} \phi_i|, \vec{\nabla} \psi_j\right>_\Omega
\end{matrix}\right.
\f\]

## How is Finite Element Method implemented in Similie ?
