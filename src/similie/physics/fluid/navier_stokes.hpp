// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstdint>

#include "cartesian_dec.hpp"

namespace sil::physics::fluid {
enum class Physics { Incompressible, Compressible, ResistiveMHD, TwoFluidPlasmaMHD };

/** Calorically perfect gas with Newtonian stress and Fourier heat conduction.
 * p=(gamma-1)(rho E-rho |u|^2/2), tau=mu(grad u+grad u^T-2/3 div(u)I),
 * q=-kappa grad T, kappa=mu cp/Pr. R=1 fixes the temperature units.
 * \important This operator and documentation is fully AI-generated.
 */
struct PerfectGas
{
    static constexpr double gamma = 1.4;
    static constexpr double prandtl = 0.71;
    static constexpr double viscosity = 1.0 / 1600.0;
    static constexpr double gas_constant = 1.0;
    static constexpr double reference_pressure = 1.0 / (gamma * 0.1 * 0.1);
    static constexpr double conductivity
            = viscosity * gamma * gas_constant / ((gamma - 1.0) * prandtl);
    KOKKOS_FUNCTION static double pressure(Field state, int i)
    {
        double kinetic = 0;
        for (int a = 0; a < 3; ++a)
            kinetic += state(a + 1, i) * state(a + 1, i) / (2 * state(0, i));
        return (gamma - 1) * (state(4, i) - kinetic);
    }
};

/** Conservative two-point kinetic-energy-preserving face transport, DEC divergence,
 * centered Newtonian stress and Fourier flux. Face-normal viscous derivatives
 * use d/h, tangential derivatives use the centered DEC operator.
 * Mass flux is mean(rho)*mean(u_n); momentum flux is F_rho*mean(u)+mean(p)n.
 * Energy flux uses mean(internal energy)+u_L dot u_R/2 and mean(p)*mean(u_n).
 * The scheme is centered and second order, without reconstruction or limiter.
 * It is not positivity preserving; callers must reject invalid states.
 * \important This operator and documentation is fully AI-generated.
 */
class CompressibleNavierStokes
{
    CartesianDEC& m_dec;
    Field m_primitive, m_gradient, m_normal_gradient, m_flux, m_difference;

public:
    explicit CompressibleNavierStokes(CartesianDEC& dec)
        : m_dec(dec)
        , m_primitive("primitive", 5, dec.cells)
        , m_gradient("gradient", 9, dec.cells)
        , m_normal_gradient("face_normal_gradient", 4, dec.cells)
        , m_flux("flux", 5, dec.cells)
        , m_difference("flux_divergence", 1, dec.cells)
    {
    }
    void primitives(Field state)
    {
        Field const primitive = m_primitive;
        Kokkos::parallel_for(
                "perfect_gas_closure",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    double const rho = state(0, i), p = PerfectGas::pressure(state, i);
                    primitive(0, i) = rho;
                    for (int a = 0; a < 3; ++a)
                        primitive(a + 1, i) = state(a + 1, i) / rho;
                    primitive(4, i) = p / (rho * PerfectGas::gas_constant);
                });
    }
    void rhs(Field state, Field result)
    {
        primitives(state);
        for (int c = 1; c < 4; ++c)
            for (int a = 0; a < 3; ++a)
                m_dec.derivative(m_primitive, c, m_gradient, 3 * (c - 1) + a, a);
        Kokkos::deep_copy(result, 0.0);
        Field const primitive = m_primitive, gradient = m_gradient, normal = m_normal_gradient,
                    flux = m_flux, difference = m_difference;
        int const n = m_dec.n;
        double const h = m_dec.h;
        for (int axis = 0; axis < 3; ++axis) {
            for (int c = 1; c < 5; ++c)
                m_dec.difference(primitive, c, normal, c - 1, axis);
            Kokkos::parallel_for(
                    "navier_stokes_face_flux",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        int const j = CartesianDEC::neighbor(i, axis, 1, n);
                        double const pl
                                = primitive(0, i) * PerfectGas::gas_constant * primitive(4, i),
                                pr = primitive(0, j) * PerfectGas::gas_constant * primitive(4, j);
                        double const ul = primitive(axis + 1, i), ur = primitive(axis + 1, j);
                        double const mass_flux
                                = 0.25 * (primitive(0, i) + primitive(0, j)) * (ul + ur);
                        double const pressure = 0.5 * (pl + pr);
                        double kinetic_cross = 0;
                        for (int a = 0; a < 3; ++a)
                            kinetic_cross += 0.5 * primitive(a + 1, i) * primitive(a + 1, j);
                        double const internal = 0.5 * PerfectGas::gas_constant
                                                * (primitive(4, i) + primitive(4, j))
                                                / (PerfectGas::gamma - 1);
                        double strain[3][3];
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                strain[a][b] = b == axis ? normal(a, i) / h
                                                         : 0.5
                                                                   * (gradient(3 * a + b, i)
                                                                      + gradient(3 * a + b, j));
                        double const divergence = strain[0][0] + strain[1][1] + strain[2][2];
                        double stress[3];
                        double work = 0;
                        for (int a = 0; a < 3; ++a) {
                            stress[a] = PerfectGas::viscosity
                                        * (strain[a][axis] + strain[axis][a]
                                           - (a == axis ? 2.0 * divergence / 3.0 : 0.0));
                            work += stress[a] * 0.5 * (primitive(a + 1, i) + primitive(a + 1, j));
                        }
                        flux(0, i) = mass_flux;
                        for (int a = 0; a < 3; ++a)
                            flux(a + 1, i)
                                    = mass_flux * 0.5 * (primitive(a + 1, i) + primitive(a + 1, j))
                                      + (a == axis ? pressure : 0.0) - stress[a];
                        flux(4, i) = mass_flux * (internal + kinetic_cross)
                                     + 0.5 * pressure * (ul + ur) - work
                                     - PerfectGas::conductivity * normal(3, i) / h;
                    });
            for (int c = 0; c < 5; ++c) {
                m_dec.difference(flux, c, difference, 0, axis, true);
                Kokkos::parallel_for(
                        "conservative_dec_divergence",
                        m_dec.cells,
                        KOKKOS_LAMBDA(int i) { result(c, i) += difference(0, i) / h; });
            }
        }
    }
};

/** Rotational incompressible Navier-Stokes, with a centered DEC gradient and
 * its negative adjoint divergence. Orthogonal periodic projection solves
 * -D G phi=-div u by matrix-free CG. The mean and even-grid checkerboard
 * null modes are untouched; divergence is in the range of this operator.
 * Viscosity uses -d^T d/h^2. Project the acceleration at every RK stage;
 * the projection potential is Bernoulli pressure p+|u|^2/2.
 * \important This operator and documentation is fully AI-generated.
 */
class IncompressibleNavierStokes
{
    CartesianDEC& m_dec;
    Field m_gradient, m_lap, m_phi, m_r, m_p, m_ap, m_temp, m_div;
    void poisson(Field input, Field output)
    {
        Kokkos::deep_copy(output, 0.0);
        Field const temp = m_temp, div = m_div;
        for (int a = 0; a < 3; ++a) {
            m_dec.derivative(input, 0, temp, 0, a);
            m_dec.derivative(temp, 0, div, 0, a);
            Kokkos::parallel_for(
                    "negative_div_gradient",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) { output(0, i) -= div(0, i); });
        }
    }
    double dot(Field left, Field right)
    {
        double value;
        Kokkos::parallel_reduce(
                "projection_dot",
                m_dec.cells,
                KOKKOS_LAMBDA(int i, double& sum) { sum += left(0, i) * right(0, i); },
                value);
        return value;
    }

public:
    Field pressure;
    std::int64_t total_iterations = 0;
    explicit IncompressibleNavierStokes(CartesianDEC& dec)
        : m_dec(dec)
        , m_gradient("velocity_gradient", 9, dec.cells)
        , m_lap("velocity_laplacian", 3, dec.cells)
        , m_phi("projection", 1, dec.cells)
        , m_r("cg_r", 1, dec.cells)
        , m_p("cg_p", 1, dec.cells)
        , m_ap("cg_ap", 1, dec.cells)
        , m_temp("cg_temp", 1, dec.cells)
        , m_div("cg_div", 1, dec.cells)
        , pressure("pressure", 1, dec.cells)
    {
        Kokkos::deep_copy(pressure, 0.0);
    }
    void rhs(Field velocity, Field result)
    {
        for (int c = 0; c < 3; ++c) {
            for (int a = 0; a < 3; ++a)
                m_dec.derivative(velocity, c, m_gradient, 3 * c + a, a);
            m_dec.laplacian(velocity, c, m_lap, c);
        }
        Field const grad = m_gradient, lap = m_lap;
        Kokkos::parallel_for(
                "rotational_momentum",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    double const omega[3]
                            = {grad(7, i) - grad(5, i),
                               grad(2, i) - grad(6, i),
                               grad(3, i) - grad(1, i)};
                    for (int a = 0; a < 3; ++a) {
                        int const b = (a + 1) % 3, c = (a + 2) % 3;
                        result(a, i) = velocity(b, i) * omega[c] - velocity(c, i) * omega[b]
                                       + PerfectGas::viscosity * lap(a, i);
                    }
                });
        project(result);
        Field const output = pressure, potential = m_phi;
        Kokkos::parallel_for(
                "recover_hydrodynamic_pressure",
                m_dec.cells,
                KOKKOS_LAMBDA(int i) {
                    double kinetic = 0;
                    for (int a = 0; a < 3; ++a)
                        kinetic += 0.5 * velocity(a, i) * velocity(a, i);
                    output(0, i) = potential(0, i) - kinetic;
                });
    }
    void project(Field velocity)
    {
        Kokkos::deep_copy(m_r, 0.0);
        Field const r = m_r, p = m_p, ap = m_ap, phi = m_phi, temp = m_temp;
        for (int a = 0; a < 3; ++a) {
            m_dec.derivative(velocity, a, temp, 0, a);
            Kokkos::parallel_for(
                    "projection_rhs",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) { r(0, i) -= temp(0, i); });
        }
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(p, r);
        double rr = dot(r, r);
        double const tolerance = 1e-20 * fmax(rr, 1.0);

        for (int iteration = 0; rr > tolerance && iteration < 4 * m_dec.n; ++iteration) {
            poisson(p, ap);
            double const pap = dot(p, ap);
            if (!(pap > 0))
                throw std::runtime_error("Incompressible projection CG breakdown");
            double const alpha = rr / pap;
            Kokkos::parallel_for(
                    "projection_cg_update",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) {
                        phi(0, i) += alpha * p(0, i);
                        r(0, i) -= alpha * ap(0, i);
                    });
            double const next = dot(r, r), beta = next / rr;
            Kokkos::parallel_for(
                    "projection_cg_direction",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) { p(0, i) = r(0, i) + beta * p(0, i); });
            rr = next;
            ++total_iterations;
        }
        if (rr > tolerance)
            throw std::runtime_error("Incompressible projection did not converge");
        for (int a = 0; a < 3; ++a) {
            m_dec.derivative(phi, 0, temp, 0, a);
            Kokkos::parallel_for(
                    "orthogonal_projection",
                    m_dec.cells,
                    KOKKOS_LAMBDA(int i) { velocity(a, i) -= temp(0, i); });
        }
    }
};
} // namespace sil::physics::fluid
