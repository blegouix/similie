// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <sstream>

#include <gtest/gtest.h>

#include "potential_flow_onelab.hpp"

namespace potential_flow = similie::onelab_interface::potential_flow_onelab;

TEST(PotentialFlow, ExtrapolatedPatchGradientsAndCirculation)
{
    std::size_t const nx = 6;
    std::size_t const ny = 3;
    std::vector<double> potential(nx * ny);
    for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i)
            potential[j * nx + i] = static_cast<double>(i * i + 2 * j);
    std::vector<std::size_t> const starts {0, 3};
    std::vector<std::size_t> const cells {3, 3};
    auto bind_domains = [](auto const& fields) {
        potential_flow::PotentialFlowPhysics const physics(
                potential_flow::FreeScalarFieldHamiltonian(0.0, 0.0, 2.0));
        sil::multidomains::Topology<2> graph(2);
        std::size_t const wall
                = graph.add_boundary({sil::tensor::NaturalScalarExtrapolationRule {}});
        graph.connect(
                {0, 0, sil::tensor::BoundarySide::Upper},
                {1, 0, sil::tensor::BoundarySide::Lower},
                false,
                1.0,
                -1.0);
        graph.connect(
                {1, 0, sil::tensor::BoundarySide::Upper},
                {0, 0, sil::tensor::BoundarySide::Lower},
                false);
        for (std::size_t id = 0; id < 2; ++id) {
            graph.connect_boundary({id, 1, sil::tensor::BoundarySide::Lower}, wall);
            graph.connect_boundary({id, 1, sil::tensor::BoundarySide::Upper}, wall);
        }
        return sil::multidomains::Multidomain(
                graph,
                std::
                        vector {sil::multidomains::domain_data(fields[0], fields[0], physics),
                                sil::multidomains::domain_data(fields[1], fields[1], physics)});
    };
    potential_flow::PotentialFlowSamples const samples = potential_flow::
            sample_potential_flow_field(potential, 5.0, nx, ny, starts, cells, bind_domains);
    for (std::size_t j = 0; j < ny; ++j)
        for (std::size_t i = 0; i < nx; ++i) {
            std::size_t const next = (i + 1) % nx;
            double const jump = i == 2 ? -5.0 : 0.0;
            double const expected_x = potential[j * nx + next] - potential[j * nx + i] + jump;
            EXPECT_DOUBLE_EQ(samples.differences[j * nx + i][0], expected_x);
            EXPECT_DOUBLE_EQ(samples.differences[j * nx + i][1], 2.0);
            if (j + 1 < ny) {
                std::array<double, 4> const& cell = samples.cell_potentials[j * nx + i];
                EXPECT_DOUBLE_EQ(cell[0], potential[j * nx + i]);
                EXPECT_DOUBLE_EQ(cell[1], potential[j * nx + next] + jump);
                EXPECT_DOUBLE_EQ(cell[2], potential[(j + 1) * nx + next] + jump);
                EXPECT_DOUBLE_EQ(cell[3], potential[(j + 1) * nx + i]);
                EXPECT_DOUBLE_EQ(cell[1] - cell[0], samples.differences[j * nx + i][0]);
            }
        }
}

TEST(PotentialFlow, CoupledCylinderSolve)
{
    std::filesystem::path const mesh_file = "potential_flow_test.msh";
    std::filesystem::path const result_file = "potential_flow_test.pos";
    // Four mapped patches, two angular cells per patch and two radial cells.
    // Keep the same physical tags and cell ordering as the ONELAB example.
    std::ofstream mesh(mesh_file);
    mesh << std::setprecision(17) << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n24\n";
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 8; ++i) {
            double const angle = i * std::acos(-1.0) / 4.0;
            mesh << 8 * j + i + 1 << ' ' << (1.0 + j) * std::cos(angle) << ' '
                 << (1.0 + j) * std::sin(angle) << " 0\n";
        }
    mesh << "$EndNodes\n$Elements\n30\n";
    std::size_t element = 1;
    for (std::size_t i = 0; i < 8; ++i)
        mesh << element++ << " 1 2 12 1 " << i + 1 << ' ' << (i + 1) % 8 + 1 << '\n';
    for (std::size_t i : {3, 4, 7, 0})
        mesh << element++ << " 1 2 " << (i == 3 || i == 4 ? 10 : 11) << " 8 " << 16 + i + 1 << ' '
             << 16 + (i + 1) % 8 + 1 << '\n';
    for (std::size_t j = 0; j < 2; ++j)
        mesh << element++ << " 1 2 13 5 " << 8 * j + 7 << ' ' << 8 * (j + 1) + 7 << '\n';
    for (std::size_t patch = 0; patch < 4; ++patch)
        for (std::size_t a = 0; a < 2; ++a)
            for (std::size_t j = 0; j < 2; ++j) {
                std::size_t const i = 2 * patch + a;
                mesh << element++ << " 3 2 " << (patch == 0 || patch == 3 ? 2 : 3) << ' '
                     << 24 + patch << ' ' << 8 * j + i + 1 << ' ' << 8 * j + (i + 1) % 8 + 1 << ' '
                     << 8 * (j + 1) + (i + 1) % 8 + 1 << ' ' << 8 * (j + 1) + i + 1 << '\n';
            }
    mesh << "$EndElements\n";
    mesh.close();
    potential_flow::Inputs inputs;
    inputs.airfoil = false;
    inputs.impose_circulation = true;
    inputs.circulation = -2.0;
    inputs.velocity = 1.0;
    inputs.box_size = 6.0;
    similie::solvers::StrongFormulationSolverSettings settings;
    settings.relative_tolerance = 1.0e-10;
    settings.max_iterations = 1000;
    potential_flow::Result const result
            = potential_flow::run(mesh_file, result_file, inputs, settings);
    EXPECT_EQ(result.domain_count, 6);
    EXPECT_EQ(result.node_count, 24);
    EXPECT_EQ(result.cell_count, 16);
    EXPECT_DOUBLE_EQ(result.circulation, -2.0);
    EXPECT_TRUE(std::isfinite(result.mass_flow_rate));
    EXPECT_TRUE(std::isfinite(result.max_speed));
    EXPECT_GT(result.max_speed, 0.0);
    EXPECT_GT(std::filesystem::file_size(result_file), 0);
    std::ifstream output(result_file);
    std::string line;
    std::vector<std::array<double, 4>> cell_potentials;
    std::getline(output, line); // Potential view header.
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
            ASSERT_TRUE(bool(std::getline(output, line)));
            std::istringstream values(line.substr(line.find("{ ") + 2));
            std::array<double, 4> potential;
            char separator;
            values >> potential[0] >> separator >> potential[1] >> separator >> potential[2]
                    >> separator >> potential[3];
            ASSERT_TRUE(bool(values));
            cell_potentials.push_back(potential);
            if (j == 1 && (i == 3 || i == 4 || i == 7 || i == 0)) {
                double const prescribed = i == 3 || i == 4 ? 0.0 : 6.0;
                EXPECT_NEAR(potential[2], prescribed, 1.0e-8);
                EXPECT_NEAR(potential[3], prescribed, 1.0e-8);
            }
        }
    std::getline(output, line); // End of potential view.
    std::getline(output, line); // Velocity view header.
    std::vector<std::array<double, 4>> vertex_pressures;
    bool varying_vertex_velocity = false;
    for (std::size_t cell = 0; cell < 16; ++cell) {
        ASSERT_TRUE(bool(std::getline(output, line)));
        std::istringstream coordinates(line.substr(3, line.find(')') - 3));
        std::array<std::array<double, 3>, 4> points;
        char separator;
        for (std::size_t a = 0; a < 4; ++a)
            for (std::size_t component = 0; component < 3; ++component) {
                coordinates >> points[a][component];
                if (a != 3 || component != 2)
                    coordinates >> separator;
            }
        ASSERT_TRUE(bool(coordinates));
        std::istringstream values(line.substr(line.find("{ ") + 2));
        std::array<double, 4> pressures;
        std::array<double, 2> first_velocity;
        for (std::size_t a = 0; a < 4; ++a) {
            double const xi = a == 1 || a == 2 ? 1.0 : 0.0;
            double const eta = a >= 2 ? 1.0 : 0.0;
            std::array<double, 2> tangent, radial;
            for (std::size_t component = 0; component < 2; ++component) {
                tangent[component] = (1.0 - eta) * (points[1][component] - points[0][component])
                                     + eta * (points[2][component] - points[3][component]);
                radial[component] = (1.0 - xi) * (points[3][component] - points[0][component])
                                    + xi * (points[2][component] - points[1][component]);
            }
            std::array<double, 4> const& potential = cell_potentials[cell];
            double const dt = (1.0 - eta) * (potential[1] - potential[0])
                              + eta * (potential[2] - potential[3]);
            double const dr = (1.0 - xi) * (potential[3] - potential[0])
                              + xi * (potential[2] - potential[1]);
            double const determinant = tangent[0] * radial[1] - tangent[1] * radial[0];
            double vx, vy, vz;
            values >> vx >> separator >> vy >> separator >> vz;
            ASSERT_TRUE(bool(values));
            EXPECT_NEAR(vx, (dt * radial[1] - tangent[1] * dr) / determinant, 1.0e-8);
            EXPECT_NEAR(vy, (tangent[0] * dr - dt * radial[0]) / determinant, 1.0e-8);
            EXPECT_DOUBLE_EQ(vz, 0.0);
            pressures[a] = -0.5 * inputs.density * (vx * vx + vy * vy);
            if (a == 0)
                first_velocity = {vx, vy};
            else
                varying_vertex_velocity
                        |= std::hypot(vx - first_velocity[0], vy - first_velocity[1]) > 1.0e-6;
            if (a != 3)
                values >> separator;
        }
        vertex_pressures.push_back(pressures);
    }
    EXPECT_TRUE(varying_vertex_velocity);
    std::getline(output, line); // End of velocity view.
    std::getline(output, line); // Pressure view header.
    for (std::array<double, 4> const& pressures : vertex_pressures) {
        ASSERT_TRUE(bool(std::getline(output, line)));
        std::istringstream values(line.substr(line.find("{ ") + 2));
        for (std::size_t a = 0; a < 4; ++a) {
            double pressure;
            values >> pressure;
            ASSERT_TRUE(bool(values));
            EXPECT_NEAR(pressure, pressures[a], 1.0e-8);
            if (a != 3) {
                char separator;
                values >> separator;
            }
        }
    }
    output.close();
    settings.use_matrix_free = false;
    potential_flow::Result const assembled
            = potential_flow::run(mesh_file, result_file, inputs, settings);
    EXPECT_NEAR(assembled.mass_flow_rate, result.mass_flow_rate, 1.0e-8);
    EXPECT_NEAR(assembled.max_speed, result.max_speed, 1.0e-8);

    auto parsed = sil::onelab_interface::gmsh::parse_supported_msh2_mesh(mesh_file);
    auto const& original = std::get<sil::onelab_interface::gmsh::QuadrilateralMesh>(parsed);
    std::ofstream scrambled(mesh_file);
    scrambled << std::setprecision(17) << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
              << original.nodes.size() << '\n';
    for (auto entry = original.nodes.rbegin(); entry != original.nodes.rend(); ++entry)
        scrambled << entry->tag << ' ' << entry->x << ' ' << entry->y << ' ' << entry->z << '\n';
    scrambled << "$EndNodes\n$Elements\n"
              << original.boundary_edges.size() + original.cells.size() << '\n';
    std::size_t number = 1;
    for (auto entry = original.cells.rbegin(); entry != original.cells.rend(); ++entry) {
        scrambled << number++ << " 3 2 " << 3 << ' ' << entry->elementary_tag + 100;
        for (std::size_t node : entry->node_tags)
            scrambled << ' ' << node;
        scrambled << '\n';
    }
    for (auto entry = original.boundary_edges.rbegin(); entry != original.boundary_edges.rend();
         ++entry)
        scrambled << number++ << " 1 2 " << entry->physical_tag << " 100 " << entry->node_tags[0]
                  << ' ' << entry->node_tags[1] << '\n';
    scrambled << "$EndElements\n";
    scrambled.close();
    potential_flow::Result const reordered
            = potential_flow::run(mesh_file, result_file, inputs, settings);
    EXPECT_EQ(reordered.domain_count, 5);
    EXPECT_NEAR(reordered.mass_flow_rate, result.mass_flow_rate, 1.e-8);
    EXPECT_NEAR(reordered.max_speed, result.max_speed, 1.e-8);
    std::filesystem::remove(mesh_file);
    std::filesystem::remove(result_file);
}
