// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>
#include <similie/physics/magnetostatics/nonlinear_magnetostatics.hpp>

#include "elasticity_onelab.hpp"

namespace elasticity = similie::onelab_interface::elasticity_onelab;

TEST(Elasticity, PlaneStressAndTensorialShear)
{
    auto const equations = similie::physics::HamiltonEquations {
            similie::physics::elasticity::LinearElasticityHamiltonian<>(200.0, 0.3)};
    similie::physics::elasticity::Strain2D const strain {.xx = 0.02, .yy = -0.01, .xy = 0.03};
    auto const stress = elasticity::detail::linear_elasticity_stress(equations, strain);
    EXPECT_NEAR(stress.xx, 200.0 / (1 - 0.3 * 0.3) * (0.02 - 0.3 * 0.01), 1e-13);
    EXPECT_NEAR(stress.yy, 200.0 / (1 - 0.3 * 0.3) * (-0.01 + 0.3 * 0.02), 1e-13);
    EXPECT_NEAR(stress.xy, 200.0 / (1 + 0.3) * 0.03, 1e-13);
}

TEST(Elasticity, ReconstructedGradientAndVirtualWork)
{
    std::array<std::array<double, 2>, 4> const positions {
            {{0.0, 0.0}, {2.0, 0.2}, {0.4, 3.0}, {2.5, 3.4}}};
    elasticity::detail::DisplacementCellReconstruction2D const reconstruction(positions);
    std::array<std::array<double, 2>, 4> nodal {};
    for (int vertex = 0; vertex < 4; ++vertex)
        nodal[vertex]
                = {0.02 * positions[vertex][0] + 0.03 * positions[vertex][1],
                   -0.01 * positions[vertex][0] + 0.04 * positions[vertex][1]};
    std::array<double, 4> const gradient = reconstruction.gradient(nodal);
    EXPECT_NEAR(gradient[0], 0.02, 1e-14);
    EXPECT_NEAR(gradient[1], 0.03, 1e-14);
    EXPECT_NEAR(gradient[2], -0.01, 1e-14);
    EXPECT_NEAR(gradient[3], 0.04, 1e-14);
    double volume = 0.0;
    for (int corner = 0; corner < 4; ++corner) {
        volume += 0.25 * reconstruction.measures[corner];
        std::array<double, 4> local_gradient {};
        for (int input = 0; input < 8; ++input)
            for (int component = 0; component < 4; ++component)
                local_gradient[component] += reconstruction.gradients[corner][input][component]
                                             * nodal[input / 2][input % 2];
        for (int component = 0; component < 4; ++component)
            EXPECT_NEAR(local_gradient[component], gradient[component], 1e-14);
    }
    EXPECT_NEAR(volume, 6.22, 1e-14);

    using Memory = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<double*, Memory> x("x", 4), y("y", 4);
    Kokkos::View<int*, Memory> active("active", 4), clamped("clamped", 4);
    auto x_host = Kokkos::create_mirror_view(x), y_host = Kokkos::create_mirror_view(y);
    for (int vertex = 0; vertex < 4; ++vertex) {
        x_host(vertex) = positions[vertex][0];
        y_host(vertex) = positions[vertex][1];
    }
    Kokkos::deep_copy(x, x_host);
    Kokkos::deep_copy(y, y_host);
    Kokkos::deep_copy(active, 1);
    Kokkos::deep_copy(clamped, 0);
    using Equations = similie::physics::HamiltonEquations<
            similie::physics::elasticity::LinearElasticityHamiltonian<>>;
    Equations const equations {
            similie::physics::elasticity::LinearElasticityHamiltonian<>(200.0, 0.3)};
    elasticity::detail::ElasticityOperator2D<Memory, Equations> const
            op(2, 2, x, y, equations, active, clamped);
    auto const data = elasticity::detail::assemble_matrix_data(op);
    std::array<std::array<double, 8>, 8> matrix {};
    for (auto const& entry : data.nonzeros)
        matrix[entry.row][entry.column] += entry.value;
    double energy = 0.0;
    for (int row = 0; row < 8; ++row)
        for (int column = 0; column < 8; ++column) {
            EXPECT_NEAR(matrix[row][column], matrix[column][row], 1e-12);
            energy += nodal[row / 2][row % 2] * matrix[row][column] * nodal[column / 2][column % 2];
        }
    similie::physics::elasticity::CauchyStress2D const stress = elasticity::detail::
            linear_elasticity_stress(equations, {.xx = 0.02, .yy = 0.04, .xy = 0.01});
    EXPECT_NEAR(energy, volume * (stress.xx * 0.02 + stress.yy * 0.04 + stress.xy * 0.02), 1e-12);
    // A free cell has exactly three zero modes: two translations and a rotation.
    // Extra zero modes would leave non-affine displacements uncontrolled.
    std::array<std::array<double, 8>, 8> eliminated = matrix;
    int rank = 0;
    for (int column = 0; column < 8; ++column) {
        int pivot = rank;
        for (int row = rank + 1; row < 8; ++row)
            if (std::abs(eliminated[row][column]) > std::abs(eliminated[pivot][column]))
                pivot = row;
        if (std::abs(eliminated[pivot][column]) < 1e-9)
            continue;
        std::swap(eliminated[rank], eliminated[pivot]);
        for (int row = rank + 1; row < 8; ++row) {
            double const factor = eliminated[row][column] / eliminated[rank][column];
            for (int entry = column; entry < 8; ++entry)
                eliminated[row][entry] -= factor * eliminated[rank][entry];
        }
        ++rank;
    }
    EXPECT_EQ(rank, 5);
    EXPECT_GT(matrix[6][6], 0.0);
    for (int row = 0; row < 8; ++row) {
        double rigid_force = 0.0;
        for (int column = 0; column < 8; ++column)
            rigid_force += matrix[row][column]
                           * (column % 2 ? positions[column / 2][0] : -positions[column / 2][1]);
        EXPECT_NEAR(rigid_force, 0.0, 1e-11);
    }
}

TEST(Elasticity, FluxBalanceAndBackendAgreement)
{
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<double*, memory_space> x("x", 9), y("y", 9);
    Kokkos::View<int*, memory_space> active("active", 9), clamped("clamped", 9);
    auto xh = Kokkos::create_mirror_view(x), yh = Kokkos::create_mirror_view(y);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            xh(i + 3 * j) = i + 0.2 * j + 0.1 * i * j;
            yh(i + 3 * j) = j + 0.3 * i;
        }
    }
    Kokkos::deep_copy(x, xh);
    Kokkos::deep_copy(y, yh);
    Kokkos::deep_copy(active, 1);
    auto const equations = similie::physics::HamiltonEquations {
            similie::physics::elasticity::LinearElasticityHamiltonian<>(200.0, 0.3)};
    elasticity::detail::ElasticityOperator2D<memory_space, decltype(equations)> const
            op(3, 3, x, y, equations, active, clamped);
    auto const data = elasticity::detail::assemble_matrix_data(op);
    std::array<std::array<double, 18>, 18> matrix {};
    for (auto const& entry : data.nonzeros) {
        matrix[entry.row][entry.column] += entry.value;
    }
    Kokkos::View<double**> u("u", 18, 1), residual("residual", 18, 1);
    auto uh = Kokkos::create_mirror_view(u);
    for (int a = 0; a < 9; ++a) {
        // A rigid rotation plus translation must have zero strain and force.
        uh(2 * a, 0) = 2 - yh(a);
        uh(2 * a + 1, 0) = 3 + xh(a);
    }
    Kokkos::deep_copy(u, uh);
    op.apply(Kokkos::DefaultExecutionSpace(), u, residual);
    auto rh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), residual);
    for (int r = 0; r < 18; ++r) {
        EXPECT_NEAR(rh(r, 0), 0, 1e-11);
    }
    for (int r = 0; r < 18; ++r) {
        uh(r, 0) = std::sin(0.7 * r);
    }
    Kokkos::deep_copy(u, uh);
    op.apply(Kokkos::DefaultExecutionSpace(), u, residual);
    Kokkos::deep_copy(rh, residual);
    double total_force_x = 0.0;
    double total_force_y = 0.0;
    for (int r = 0; r < 18; ++r) {
        double expected = 0.0;
        for (int c = 0; c < 18; ++c)
            expected += matrix[r][c] * uh(c, 0);
        EXPECT_NEAR(rh(r, 0), expected, 1e-11);
        if (r % 2 == 0)
            total_force_x += rh(r, 0);
        else
            total_force_y += rh(r, 0);
    }
    // Every internal dual face contributes equal and opposite nodal forces.
    EXPECT_NEAR(total_force_x, 0.0, 1e-11);
    EXPECT_NEAR(total_force_y, 0.0, 1e-11);

    // A homogeneous affine displacement gives constant stress and therefore
    // zero integrated divergence in the interior control volume.
    for (int a = 0; a < 9; ++a) {
        uh(2 * a, 0) = 0.02 * xh(a) + 0.03 * yh(a);
        uh(2 * a + 1, 0) = -0.01 * xh(a) + 0.04 * yh(a);
    }
    Kokkos::deep_copy(u, uh);
    op.apply(Kokkos::DefaultExecutionSpace(), u, residual);
    Kokkos::deep_copy(rh, residual);
    EXPECT_NEAR(rh(2 * 4, 0), 0.0, 1e-11);
    EXPECT_NEAR(rh(2 * 4 + 1, 0), 0.0, 1e-11);
}

TEST(Elasticity, ReconstructedEnergyConvergesUnderRefinement)
{
    using Memory = Kokkos::DefaultExecutionSpace::memory_space;
    using Equations = similie::physics::HamiltonEquations<
            similie::physics::elasticity::LinearElasticityHamiltonian<>>;
    Equations const equations {
            similie::physics::elasticity::LinearElasticityHamiltonian<>(200.0, 0.3)};
    // On the unit square, u=(xy,0) gives grad u=(y,x;0,0).
    // The exact integrated virtual work is (C_xxxx + mu)/3.
    double const exact = (elasticity::detail::linear_elasticity_stress(equations, {.xx = 1.0}).xx
                          + elasticity::detail::linear_elasticity_stress(equations, {.xy = 0.5}).xy)
                         / 3.0;
    double previous_error = 0.0;
    for (int cells : {2, 4}) {
        int const n = cells + 1;
        Kokkos::View<double*, Memory> x("x", n * n), y("y", n * n);
        Kokkos::View<int*, Memory> active("active", n * n), clamped("clamped", n * n);
        auto x_host = Kokkos::create_mirror_view(x), y_host = Kokkos::create_mirror_view(y);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                x_host(i + n * j) = double(i) / cells;
                y_host(i + n * j) = double(j) / cells;
            }
        Kokkos::deep_copy(x, x_host);
        Kokkos::deep_copy(y, y_host);
        Kokkos::deep_copy(active, 1);
        Kokkos::deep_copy(clamped, 0);
        elasticity::detail::ElasticityOperator2D<Memory, Equations> const
                op(n, n, x, y, equations, active, clamped);
        auto const matrix = elasticity::detail::assemble_matrix_data(op);
        double work = 0.0;
        for (auto const& entry : matrix.nonzeros)
            if (entry.row % 2 == 0 && entry.column % 2 == 0)
                work += x_host(entry.row / 2) * y_host(entry.row / 2) * entry.value
                        * x_host(entry.column / 2) * y_host(entry.column / 2);
        double const error = std::abs(work - exact);
        EXPECT_GT(work, 0.0);
        if (previous_error > 0.0)
            EXPECT_NEAR(error, previous_error / 4.0, 1e-10);
        previous_error = error;
    }
}

TEST(MagneticEnergy, DerivativeMatchesNonlinearConstitutiveLaw)
{
    similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<4> const
            curve(std::array<double, 4> {0.0, 1.0, 2.0, 3.0},
                  std::array<double, 4> {0.0, 2.0, 7.0, 20.0});
    EXPECT_DOUBLE_EQ(curve.magnetic_energy_from_q(0.0), 0.0);
    // Include the extrapolated interval as well as every tabulated segment.
    for (double b : {0.25, 1.25, 2.5, 3.5}) {
        double const step = 1e-5;
        double const derivative = (curve.magnetic_energy_from_q((b + step) * (b + step))
                                   - curve.magnetic_energy_from_q((b - step) * (b - step)))
                                  / (2 * step);
        EXPECT_NEAR(derivative, curve.h_from_b(b), 1e-8);
    }
}
