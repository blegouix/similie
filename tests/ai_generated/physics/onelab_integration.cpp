// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <gtest/gtest.h>

#include "magnetostatics_onelab.hpp"

namespace magnetostatics = similie::onelab_interface::magnetostatics_onelab;

namespace {

template <class Grid>
void check_diagnostics(Grid const& grid, double depth, std::size_t face_count)
{
    magnetostatics::Inputs inputs;
    inputs.length_z = depth;
    inputs.positive_electrical_conductor_tags = {1};
    inputs.diagnostic_region_tags = {1};
    std::vector<magnetostatics::detail::CellInputFields> cell_inputs(grid.ordered_cells.size());
    std::vector<magnetostatics::detail::CellPostProcessFields> cell_outputs(
            grid.ordered_cells.size());
    for (magnetostatics::detail::CellInputFields& input : cell_inputs)
        input.current_density = {0.0, 0.0, 9.0};
    for (magnetostatics::detail::CellPostProcessFields& output : cell_outputs) {
        output.magnetic_induction = {2.0, 3.0, 4.0};
        output.magnetic_field = {5.0, 7.0, 11.0};
    }
    magnetostatics::Result result;
    magnetostatics::detail::integrate_diagnostics(grid, inputs, cell_inputs, cell_outputs, result);
    EXPECT_EQ(result.num_current_cells, 4);
    EXPECT_EQ(result.num_diagnostic_faces, face_count);
    EXPECT_DOUBLE_EQ(result.diagnostic_current_integral, 270.0);
    EXPECT_DOUBLE_EQ(result.diagnostic_surface_measure, 6.0 * depth);
    EXPECT_DOUBLE_EQ(result.diagnostic_flux_integral, 18.0 * depth);
    EXPECT_DOUBLE_EQ(result.diagnostic_traction_integral[0], 84.0 * depth);
    EXPECT_DOUBLE_EQ(result.diagnostic_traction_integral[1], -99.0 * depth);
    EXPECT_DOUBLE_EQ(result.diagnostic_traction_integral[2], 168.0 * depth);
    EXPECT_NEAR(
            result.diagnostic_traction_magnitude_integral,
            6.0 * depth * std::sqrt(14.0 * 14.0 + 16.5 * 16.5 + 28.0 * 28.0),
            1e-11);

    inputs.positive_electrical_conductor_tags.clear();
    inputs.diagnostic_region_tags.clear();
    magnetostatics::detail::integrate_diagnostics(grid, inputs, cell_inputs, cell_outputs, result);
    EXPECT_EQ(result.num_current_cells, 0);
    EXPECT_EQ(result.num_diagnostic_faces, 0);
    EXPECT_DOUBLE_EQ(result.diagnostic_current_integral, 0.0);
    EXPECT_DOUBLE_EQ(result.diagnostic_surface_measure, 0.0);
    EXPECT_DOUBLE_EQ(result.diagnostic_flux_integral, 0.0);
    EXPECT_DOUBLE_EQ(result.diagnostic_traction_integral[0], 0.0);
    EXPECT_DOUBLE_EQ(result.diagnostic_traction_integral[1], 0.0);
    EXPECT_DOUBLE_EQ(result.diagnostic_traction_integral[2], 0.0);
}

TEST(OnelabIntegration, PlanarExtrusionAndCurrentSection)
{
    sil::onelab_interface::gmsh::StructuredGrid2D grid;
    grid.x_coords = {-2.0, 1.0, 4.0};
    grid.y_coords = {0.0, 2.0, 5.0};
    grid.z_value = 7.0;
    grid.ordered_cells.resize(4);
    for (sil::onelab_interface::gmsh::QuadrilateralCell& cell : grid.ordered_cells)
        cell.physical_tag = 1;
    check_diagnostics(grid, 2.0, 2);
}

TEST(OnelabIntegration, SpatialSurfaceAndMiddleCurrentSection)
{
    sil::onelab_interface::gmsh::StructuredGrid3D grid;
    grid.x_coords = {-2.0, 1.0, 4.0};
    grid.y_coords = {0.0, 2.0, 5.0};
    grid.z_coords = {1.0, 2.0, 5.0};
    grid.ordered_cells.resize(8);
    for (sil::onelab_interface::gmsh::HexahedralCell& cell : grid.ordered_cells)
        cell.physical_tag = 1;
    check_diagnostics(grid, 4.0, 4);
}

TEST(OnelabIntegration, NonlinearMagneticJacobianOnNonuniformGrid)
{
    using namespace magnetostatics::detail::magnetostatics_local;
    using Curve = similie::physics::magnetostatics::InterpolatedNonlinearBHCurve<4>;
    using Equations = similie::physics::HamiltonEquations<
            similie::physics::magnetostatics::NonlinearMagnetostaticsHamiltonian<Curve>>;
    Curve const
            curve(std::array<double, 4> {0.0, 0.4, 1.3, 3.0},
                  std::array<double, 4> {0.0, 0.5, 3.0, 10.0});
    Equations const equations {
            similie::physics::magnetostatics::NonlinearMagnetostaticsHamiltonian<Curve>(curve)};
    Kokkos::View<double*, Kokkos::HostSpace> x("x", 5), y("y", 5);
    for (int i = 0; i < 5; ++i) {
        x(i) = 0.15 * i + 0.1 * i * i;
        y(i) = 0.25 * i + 0.05 * i * i;
    }
    MagnetostaticsOperator2D<Kokkos::HostSpace, Equations, PotentialCoboundary2D> const
            op(equations, x, y, similie::solvers::Criterion::PotentialTemporalDerivative);
    Kokkos::View<double**, Kokkos::HostSpace> state("state", 25, 1), direction("direction", 25, 1),
            upper("upper", 25, 1), lower("lower", 25, 1), upper_response("upper_response", 25, 1),
            lower_response("lower_response", 25, 1), tangent("tangent", 25, 1);
    double const step = 1e-6;
    for (int row = 0; row < 25; ++row) {
        state(row, 0) = 0.35 * std::sin(0.4 * row) + 0.02 * row;
        direction(row, 0) = 0.1 * std::cos(0.7 * row);
        upper(row, 0) = state(row, 0) + step * direction(row, 0);
        lower(row, 0) = state(row, 0) - step * direction(row, 0);
    }
    op.apply(Kokkos::DefaultHostExecutionSpace(), upper, upper_response);
    op.apply(Kokkos::DefaultHostExecutionSpace(), lower, lower_response);
    apply_jacobian(Kokkos::DefaultHostExecutionSpace(), op, state, direction, tangent);
    Kokkos::fence();
    for (int row = 0; row < 25; ++row)
        EXPECT_NEAR(
                tangent(row, 0),
                (upper_response(row, 0) - lower_response(row, 0)) / (2 * step),
                2e-7);
}

TEST(OnelabIntegration, MaxwellStressDivergenceUsesPhysicalUnits)
{
    using namespace magnetostatics::detail::magnetostatics_local;
    ddc::DiscreteDomain<DDimX, DDimY> const
            domain(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                   ddc::DiscreteVector<DDimX, DDimY>(5, 5));
    // A linear stress has the same physical divergence at every mesh scale.
    for (std::array<double, 2> const spacing :
         {std::array<double, 2> {2.0, 3.0}, std::array<double, 2> {0.2, 0.7}}) {
        auto position = [&](ddc::DiscreteElement<DDimX, DDimY> const& elem) {
            return std::array<double, 2> {
                    spacing[0] * ddc::DiscreteElement<DDimX>(elem).uid(),
                    spacing[1] * ddc::DiscreteElement<DDimY>(elem).uid()};
        };
        magnetostatics::detail::fill_force_density_on_cell_domain(
                domain,
                [&](ddc::DiscreteElement<DDimX, DDimY> const& elem) {
                    std::array<double, 2> const point = position(elem);
                    return std::array<double, 6> {
                            2.0 * point[0],
                            3.0 * point[1],
                            0.0,
                            5.0 * point[0] + 7.0 * point[1],
                            11.0 * point[0],
                            13.0 * point[1]};
                },
                position,
                [&](ddc::DiscreteElement<DDimX, DDimY> const& elem,
                    std::array<double, 3> const& force) {
                    std::size_t const i = ddc::DiscreteElement<DDimX>(elem).uid();
                    std::size_t const j = ddc::DiscreteElement<DDimY>(elem).uid();
                    if (i > 0 && i < 4 && j > 0 && j < 4) {
                        EXPECT_NEAR(force[0], 9.0, 1e-11);
                        EXPECT_NEAR(force[1], 8.0, 1e-11);
                        EXPECT_NEAR(force[2], 24.0, 1e-11);
                    }
                });
    }
}

TEST(OnelabIntegration, HexahedralMaxwellStressUsesAllSpatialDerivatives)
{
    for (std::array<double, 3> const spacing :
         {std::array<double, 3> {2.0, 3.0, 0.4}, std::array<double, 3> {0.2, 0.7, 1.3}}) {
        sil::onelab_interface::gmsh::StructuredGrid3D grid;
        for (int i = 0; i < 6; ++i) {
            grid.x_coords.push_back(spacing[0] * i);
            grid.y_coords.push_back(spacing[1] * i);
            grid.z_coords.push_back(spacing[2] * i);
        }
        std::vector<magnetostatics::detail::CellPostProcessFields> output(125);
        // Exercise each axis independently, then their sum and a constant stress.
        // Packed stress order is xx, yy, zz, xy, xz, yz.
        for (int axis = 0; axis < 5; ++axis) {
            SCOPED_TRACE(
                    ::testing::Message() << "axis=" << axis << ", spacing=" << spacing[0] << ","
                                         << spacing[1] << "," << spacing[2]);
            double const dx = axis == 0 || axis == 3 ? 1.0 : 0.0;
            double const dy = axis == 1 || axis == 3 ? 1.0 : 0.0;
            double const dz = axis == 2 || axis == 3 ? 1.0 : 0.0;
            for (std::size_t k = 0; k < 5; ++k)
                for (std::size_t j = 0; j < 5; ++j)
                    for (std::size_t i = 0; i < 5; ++i) {
                        double const x = grid.cell_center_x(i);
                        double const y = grid.cell_center_y(j);
                        double const z = grid.cell_center_z(k);
                        output[grid.cell_index(i, j, k)].maxwell_stress
                                = {31.0 + dx * 2.0 * x,
                                   37.0 + dy * 3.0 * y,
                                   41.0 + dz * 5.0 * z,
                                   43.0 + dx * 7.0 * x + dy * 11.0 * y,
                                   47.0 + dx * 13.0 * x + dz * 17.0 * z,
                                   53.0 + dy * 19.0 * y + dz * 23.0 * z};
                    }
            magnetostatics::detail::fill_force_density_on_hexahedral_grid(grid, output);
            for (std::size_t k = 1; k < 4; ++k)
                for (std::size_t j = 1; j < 4; ++j)
                    for (std::size_t i = 1; i < 4; ++i) {
                        std::array<double, 3> const& force
                                = output[grid.cell_index(i, j, k)].force_density;
                        EXPECT_NEAR(force[0], 2.0 * dx + 11.0 * dy + 17.0 * dz, 1e-10);
                        EXPECT_NEAR(force[1], 7.0 * dx + 3.0 * dy + 23.0 * dz, 1e-10);
                        EXPECT_NEAR(force[2], 13.0 * dx + 19.0 * dy + 5.0 * dz, 1e-10);
                    }
        }
    }
}

} // namespace
