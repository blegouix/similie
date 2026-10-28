// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/magnetic_form_response.hpp>
#include <similie/physics/magnetostatics/nonlinear_magnetostatics.hpp>

namespace {
struct X
{
};
struct Y
{
};
struct Z
{
};
using Curve = similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<4>;
using Equations = similie::physics::HamiltonEquations<
        similie::physics::magnetostatics::NonlinearMagnetostaticsHamiltonian<Curve>>;

struct MagneticChecks
{
    Kokkos::View<double*> errors;
    Curve curve;

    KOKKOS_FUNCTION void operator()(int) const
    {
        using MetricIndex = sil::tensor::TensorSymmetricIndex<
                sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
                sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
        std::array<double, MetricIndex::mem_size()> storage {};
        sil::tensor::Tensor metric(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<MetricIndex>,
                        Kokkos::layout_right,
                        Kokkos::DefaultExecutionSpace::memory_space>(
                        storage.data(),
                        sil::tensor::TensorAccessor<MetricIndex>::domain()));
        // Pullback of the Euclidean metric by x=(2u+v,3v,4w).
        metric(metric.template access_element<X, X>()) = 4.0;
        metric(metric.template access_element<X, Y>()) = 2.0;
        metric(metric.template access_element<Y, Y>()) = 10.0;
        metric(metric.template access_element<Z, Z>()) = 16.0;
        similie::physics::magnetostatics::MagneticFormResponse<X, Y, Z> const
                response(metric, ddc::DiscreteElement<>());
        std::array<double, 3> const induction {8.0, -3.0, 6.0};
        std::array<double, 3> const dual {26.0 / 24.0, -14.0 / 24.0, 96.0 / 24.0};
        double field_error = 0.0;
        std::array<double, 3> const linear = response.field(induction, 0.5);
        for (int component = 0; component < 3; ++component)
            field_error += Kokkos::abs(linear[component] - 0.5 * dual[component]);

        Equations const equations {
                similie::physics::magnetostatics::NonlinearMagnetostaticsHamiltonian<Curve>(curve)};
        Equations::State state {};
        state.moments = induction;
        double const q = 826.0 / (24.0 * 24.0);
        double const norm = Kokkos::sqrt(q);
        double const reluctivity = curve.h_from_b(norm) / norm;
        std::array<double, 3> const field = response.field(equations, state, 0);
        std::array<double, 9> const tangent = response.jacobian(equations, state, 0);
        for (int component = 0; component < 3; ++component)
            field_error += Kokkos::abs(field[component] - reluctivity * dual[component]);
        double tangent_error = 0.0, energy_error = 0.0;
        double const step = 1.e-5;
        for (int column = 0; column < 3; ++column) {
            Equations::State upper = state, lower = state;
            upper.moments[column] += step;
            lower.moments[column] -= step;
            std::array<double, 3> const plus = response.field(equations, upper, 0);
            std::array<double, 3> const minus = response.field(equations, lower, 0);
            for (int row = 0; row < 3; ++row)
                tangent_error += Kokkos::abs(
                        tangent[3 * row + column] - (plus[row] - minus[row]) / (2.0 * step));
            // The Hamiltonian top form is W(|B|_g^2) sqrt(det g) du dv dw.
            double const delta_q = 2.0 * dual[column] * step / 24.0;
            double const diagonal = column == 0 ? 4.0 : column == 1 ? 10.0 : 16.0;
            double const quadratic = diagonal * step * step / (24.0 * 24.0);
            double const energy_derivative
                    = 24.0
                      * (curve.magnetic_energy_from_q(q + delta_q + quadratic)
                         - curve.magnetic_energy_from_q(q - delta_q + quadratic))
                      / (2.0 * step);
            energy_error += Kokkos::abs(field[column] - energy_derivative);
        }
        Equations::State const zero {};
        std::array<double, 3> const zero_field = response.field(equations, zero, 0);
        std::array<double, 9> const zero_tangent = response.jacobian(equations, zero, 0);
        field_error += Kokkos::abs(zero_field[0]) + Kokkos::abs(zero_field[1])
                       + Kokkos::abs(zero_field[2]);
        tangent_error += Kokkos::abs(zero_tangent[1] - 1.25 * 2.0 / 24.0);

        similie::physics::magnetostatics::NonlinearMagneticInductionToMagneticField<Curve> const
                law(curve);
        std::array<double, 3> const constitutive
                = response.constitutive_field(law, typename decltype(law)::State {induction}, 0);
        for (int component = 0; component < 3; ++component)
            field_error += Kokkos::abs(constitutive[component] - field[component]);
        errors(0) = field_error;
        errors(1) = tangent_error;
        errors(2) = energy_error;
    }
};

TEST(MagneticForms, NonDiagonalMetricConstitutiveLawTangentAndEnergy)
{
    Curve const
            curve(std::array<double, 4> {0.0, 0.4, 1.3, 3.0},
                  std::array<double, 4> {0.0, 0.5, 3.0, 10.0});
    Kokkos::View<double*> errors("magnetic_form_errors", 3);
    Kokkos::parallel_for("magnetic_form_checks", 1, MagneticChecks {errors, curve});
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    EXPECT_LT(host(0), 1.e-12);
    EXPECT_LT(host(1), 1.e-8);
    EXPECT_LT(host(2), 1.e-8);
}
} // namespace
