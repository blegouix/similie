# Finite Element module {#finite_element_module}
<!--
SPDX-FileCopyrightText: 2025 Baptiste Legouix
SPDX-License-Identifier: GPL-3.0-or-later
-->

## Why partial differential equations are importants for physics ?

Given a multidimensional function \f$u:x\rightarrow \; .\f$, any [partial differential equation](https://en.wikipedia.org/wiki/Partial_differential_equation) can be written:

\f\[
\mathcal{F}(D^k u, D^{k-1} u,... ,D u, u, x) = f
\f\]

Where \f$D^k\f$ is a \f$k-\f$order differential operator. Any partial differential equation can be splitted into a Hamiltonian and non-Hamiltonian parts:

\f\[
\frac{\partial u}{\partial t} + \{\mathcal{H}, u\} + \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x) = f
\f\]

Where \f$\{\mathcal{H}, u\}\f$ is the Poisson bracket between the Hamiltonian \f$\mathcal{H}\f$ of the system and \f$u\f$. It can always be written \f$\{\mathcal{H}, u\} = \mathcal{J}\frac{\partial \mathcal{H}}{\partial x}\f$ in term of the symplectic matrix \f$\mathcal{J}\f$. The Hamiltonian part of the PDE is associated to conserved quantities such as mass, momentum or energy.

\note If a PDE involves a second-order temporal derivative \f$\frac{\partial^2 u}{\partial t^2}\f$ (like in wave equation) we can always define \f$U = (\frac{\partial u}{\partial t}, u)\f$ and write the PDE in term of  \f$U\f$.

Partial differential equations are the main mathematical tool to describe rules of classical physics.

## What is Finite Element Method ?

Finite Element Method aims to discretize PDE to produce numerical schemes to solve it. It relies on the weak formulation of the PDE:

\f\[
\int \frac{\partial u}{\partial t}v\; d\Omega + \int \mathcal{J}\mathcal{H}\frac{\partial v}{\partial x} \; d\Omega + \int \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x) v \; d\Omega = \int f v \; d\Omega
\f\]

\note \f$\int \mathcal{J}\frac{\partial \mathcal{H}}{\partial x}v \; d\Omega\f$ has been transformed into \f$\int \mathcal{J}\mathcal{H}\frac{\partial v}{\partial x} \; d\Omega\f$ by part integration. 

Where \f$v\f$ is a test function belonging to a basis of test functions.

We decompose \f$u\f$ and \f$v\f$ in a discrete basis function \f$\phi\f$ (we consider both \f$u\f$ and \f$v\f$ to be approximated in the same discrete basis function but in principle we could distinguish two different basis function). 

\f\[
\left\{\begin{matrix}
u = \sum_i \tilde{u}_i \phi_i\\
v = \sum_i \tilde{v}_i \phi_i
\end{matrix}\right.
\f\]

Leading to:

\f\[
\int \frac{\partial \tilde{u}_i}{\partial t}\tilde{v}_j\phi_i\phi_j\; d\Omega + \int \mathcal{J}\mathcal{H}\tilde{v}_j\frac{\partial \phi_j}{\partial x} \; d\Omega + \int \mathcal{D}(D^k u, D^{k-1} u,... ,D u, u, x) v \; d\Omega = \int \tilde{f}_i \tilde{v}_j \phi_i\phi_j\; d\Omega
\f\]
