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

} // namespace
