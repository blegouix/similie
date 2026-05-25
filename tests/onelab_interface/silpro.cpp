// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <similie/physics/dedonder_weyl_equations.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/scalar_field/scalar_field_with_power_coupling.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include "linear_magnetostatics_onelab.hpp"
#include "onelab_interface.hpp"

namespace {

std::filesystem::path test_file(char const* name)
{
    return std::filesystem::path(SIMILIE_ONELAB_TEST_DIR) / name;
}

} // namespace

TEST(OnelabInterface, ParseLinearMagnetostaticsSilpro)
{
    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(
                    test_file("magnetostatics.silpro"));

    EXPECT_EQ(problem.name, "Test magnetostatics problem");
    EXPECT_EQ(problem.physics, similie::onelab_interface::SupportedPhysics::LinearMagnetostatics);
    EXPECT_EQ(
            problem.solver,
            similie::onelab_interface::SupportedSolver::MinimizeStrongFormulationResidual);
    EXPECT_EQ(problem.solver_settings.max_iterations, 4321U);
    EXPECT_DOUBLE_EQ(problem.solver_settings.relative_tolerance, 1e-8);
    EXPECT_EQ(problem.solver_settings.jacobi_max_block_size, 3U);
    EXPECT_FALSE(problem.solver_settings.use_matrix_free);
    EXPECT_EQ(
            problem.solver_settings.criterion,
            similie::solvers::Criterion::MomentsTemporalDerivative);
    EXPECT_EQ(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .name,
            "SingleElectricalConductorMaterialWithSingleLinearMagneticMaterial");
    EXPECT_EQ(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .positive_electrical_conductor_tags.size(),
            1U);
    EXPECT_EQ(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .negative_electrical_conductor_tags.size(),
            1U);
    EXPECT_EQ(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .magnetic_material_tags.size(),
            2U);
    EXPECT_FALSE(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .use_nonlinear_magnetic_material);
    EXPECT_EQ(problem.force_density_diagnostics_postprocess.name, "ForceDensityDiagnostics");
    EXPECT_EQ(problem.force_density_diagnostics_postprocess.diagnostic_region_tags.size(), 1U);
}

TEST(OnelabInterface, ParseScalarFieldSilpro)
{
    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(
                    test_file("scalar_field.silpro"));

    EXPECT_EQ(problem.name, "Test scalar field problem");
    EXPECT_EQ(
            problem.physics,
            similie::onelab_interface::SupportedPhysics::ScalarFieldWithPowerCoupling);
    EXPECT_EQ(problem.solver_settings.max_iterations, 250U);
    EXPECT_DOUBLE_EQ(problem.solver_settings.relative_tolerance, 1e-6);
    EXPECT_TRUE(problem.solver_settings.use_matrix_free);
    EXPECT_EQ(
            problem.solver_settings.criterion,
            similie::solvers::Criterion::MomentsTemporalDerivative);
    EXPECT_DOUBLE_EQ(problem.scalar_field.mass, 2.5);
    EXPECT_DOUBLE_EQ(problem.scalar_field.coupling_constant, 3.25);
    EXPECT_DOUBLE_EQ(problem.scalar_field.coupling_power, 6.0);
}

TEST(OnelabInterface, ParseNonLinearMagnetostaticsSilpro)
{
    std::filesystem::path const file = std::filesystem::temp_directory_path()
                                       / "similie_nonlinear_magnetostatics_test.silpro";
    {
        std::ofstream output(file);
        output << "Problem {\n"
                  "  Name \"Nonlinear magnetostatics\";\n"
                  "  Physics NonLinearMagnetostatics;\n"
                  "  Solver MinimizeStrongFormulationResidual;\n"
                  "}\n"
                  "Solver {\n"
                  "  Criterion MomentsTemporalDerivative;\n"
                  "}\n"
                  "Preprocess {\n"
                  "  NonlinearBHCurve EIcore;\n"
                  "}\n";
    }

    similie::onelab_interface::SilproProblem const problem
            = similie::onelab_interface::OnelabInterface::parse_silpro_file(file);
    EXPECT_EQ(
            problem.physics,
            similie::onelab_interface::SupportedPhysics::NonLinearMagnetostatics);
    EXPECT_EQ(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .nonlinear_bh_curve,
            "EIcore");

    std::filesystem::remove(file);
}

TEST(OnelabInterface, HamiltonEquationsStaticPotentialDerivative)
{
    using similie::physics::HamiltonEquations;
    using similie::physics::magnetostatics::LinearMagnetostaticsHamiltonian;
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;

    using memory_space = Kokkos::HostSpace;
    scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>(
                    ddc::DiscreteDomain<DDimX, DDimY>(
                            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                            ddc::DiscreteVector<DDimX, DDimY>(1, 1)),
                    sil::tensor::TensorAccessor<ScalarPotentialIndex>().domain()),
            ddc::KokkosAllocator<double, memory_space>());
    ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    mu_host(0, 0, 0) = 2.0;
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);

    auto const hamiltonian = LinearMagnetostaticsHamiltonian(mu_tensor);
    auto const equations = HamiltonEquations {hamiltonian};
    auto const elem = ddc::DiscreteElement<DDimX, DDimY>(0, 0);
    std::array<double, 3> const moments {4.0, 6.0, 8.0};

    EXPECT_DOUBLE_EQ(
            equations.template dpotential_dt<0>(std::span<double const, 3>(moments), elem),
            2.0);
    EXPECT_DOUBLE_EQ(
            equations.template dpotential_dt<1>(std::span<double const, 3>(moments), elem),
            3.0);
    EXPECT_DOUBLE_EQ(
            equations.template dpotential_dt<2>(std::span<double const, 3>(moments), elem),
            4.0);
}

TEST(OnelabInterface, HamiltonEquationsStaticMomentumDerivative)
{
    using similie::physics::HamiltonEquations;
    using similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian;

    HamiltonEquations const equations(ScalarFieldWithPowerCouplingHamiltonian(2.0, 0.0, 4.0));
    std::array<double, 1> const potential {3.0};

    EXPECT_DOUBLE_EQ(
            equations.template dmoments_dt<0>(std::span<double const, 1>(potential)),
            12.0);
    EXPECT_DOUBLE_EQ(
            equations.template dmoments_dt<1>(std::span<double const, 1>(potential)),
            12.0);
}

TEST(OnelabInterface, DeDonderWeylEquationsStaticInterfaces)
{
    using similie::physics::DeDonderWeylEquations;
    using similie::physics::scalar_field::ScalarFieldWithPowerCouplingHamiltonian;

    DeDonderWeylEquations const equations(ScalarFieldWithPowerCouplingHamiltonian(2.0, 0.0, 4.0));
    std::array<double, 3> const moments {5.0, 5.0, 5.0};
    std::array<double, 1> const potential {3.0};

    EXPECT_DOUBLE_EQ(
            equations.template potential_grad<0>(std::span<double const, 3>(moments)),
            -5.0);
    EXPECT_DOUBLE_EQ(
            equations.template potential_grad<1>(std::span<double const, 3>(moments)),
            5.0);
    EXPECT_DOUBLE_EQ(equations.moments_div(std::span<double const, 1>(potential)), 12.0);
}

TEST(OnelabInterface, LinearMagneticInductionToMagneticFieldValueAndApplication)
{
    using similie::physics::magnetostatics::LinearMagneticInductionToMagneticField;

    LinearMagneticInductionToMagneticField const constitutive_law(2.0);
    EXPECT_DOUBLE_EQ(constitutive_law.value(3.0, 4.0), 1.5);
    EXPECT_DOUBLE_EQ(constitutive_law.jacobian(3.0, 4.0), 1.5);
    EXPECT_DOUBLE_EQ(constitutive_law(3.0, 4.0), 6.0);
    EXPECT_DOUBLE_EQ(constitutive_law.inverse_value(3.0, 5.0), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(constitutive_law.inverse(3.0, 5.0), 10.0 / 3.0);
}

TEST(OnelabInterface, MagnetostaticsPostProcessInductionUsesLibraryStencil)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::
            fill_post_process_fields_on_cell_domain;

    auto const node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(3, 3));
    auto const cell_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(2, 2));
    std::vector<double> const x_coords {0.0, 1.0, 2.0};
    std::vector<double> const y_coords {0.0, 1.0, 2.0};
    auto node_value_z
            = [&](std::size_t i, std::size_t j) { return 3.0 * x_coords[i] - 2.0 * y_coords[j]; };

    std::array<double, 3> induction {};
    fill_post_process_fields_on_cell_domain(
            cell_domain,
            node_domain,
            [&](auto node_elem) {
                std::size_t const i
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(node_elem).uid());
                std::size_t const j
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(node_elem).uid());
                return std::array<double, 2> {
                        x_coords[i],
                        y_coords[j],
                };
            },
            [](auto) { return 1.0; },
            [](auto) { return false; },
            [](auto, auto) { return std::array<double, 3> {0.0, 0.0, 0.0}; },
            node_value_z,
            [&](auto elem, std::array<double, 3> const& value, std::array<double, 3> const&) {
                if (ddc::DiscreteElement<DDimX>(elem).uid() == 0
                    && ddc::DiscreteElement<DDimY>(elem).uid() == 0) {
                    induction = value;
                }
            });

    EXPECT_NEAR(induction[0], -2.0, 1e-12);
    EXPECT_NEAR(induction[1], -3.0, 1e-12);
    EXPECT_DOUBLE_EQ(induction[2], 0.0);
}

TEST(OnelabInterface, MagnetostaticsPostProcessInductionIsCellCentered)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::
            fill_post_process_fields_on_cell_domain;

    auto const node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(3, 3));
    auto const cell_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(2, 2));
    std::vector<double> const x_coords {0.0, 1.0, 2.0};
    std::vector<double> const y_coords {0.0, 1.0, 2.0};
    auto node_value_z = [&](std::size_t i, std::size_t j) {
        return x_coords[i] * x_coords[i] - y_coords[j] * y_coords[j];
    };

    std::array<double, 3> induction {};
    fill_post_process_fields_on_cell_domain(
            cell_domain,
            node_domain,
            [&](auto node_elem) {
                std::size_t const i
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(node_elem).uid());
                std::size_t const j
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(node_elem).uid());
                return std::array<double, 2> {
                        x_coords[i],
                        y_coords[j],
                };
            },
            [](auto) { return 1.0; },
            [](auto) { return false; },
            [](auto, auto) { return std::array<double, 3> {0.0, 0.0, 0.0}; },
            node_value_z,
            [&](auto elem, std::array<double, 3> const& value, std::array<double, 3> const&) {
                if (ddc::DiscreteElement<DDimX>(elem).uid() == 0
                    && ddc::DiscreteElement<DDimY>(elem).uid() == 0) {
                    induction = value;
                }
            });

    EXPECT_NEAR(induction[0], -1.0, 1e-12);
    EXPECT_NEAR(induction[1], -1.0, 1e-12);
    EXPECT_DOUBLE_EQ(induction[2], 0.0);
}

TEST(OnelabInterface, MagnetostaticsPostProcessInductionCentersEachComponentOnItsOwnFaces)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::
            fill_post_process_fields_on_cell_domain;

    auto const node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(3, 3));
    auto const cell_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(2, 2));
    std::vector<double> const x_coords {0.0, 1.0, 2.0};
    std::vector<double> const y_coords {0.0, 1.0, 2.0};
    auto node_value_z = [&](std::size_t i, std::size_t j) {
        return x_coords[i] * x_coords[i] + 2.0 * y_coords[j] * y_coords[j];
    };

    std::array<double, 3> induction {};
    fill_post_process_fields_on_cell_domain(
            cell_domain,
            node_domain,
            [&](auto node_elem) {
                std::size_t const i
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(node_elem).uid());
                std::size_t const j
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(node_elem).uid());
                return std::array<double, 2> {
                        x_coords[i],
                        y_coords[j],
                };
            },
            [](auto) { return 1.0; },
            [](auto) { return false; },
            [](auto, auto) { return std::array<double, 3> {0.0, 0.0, 0.0}; },
            node_value_z,
            [&](auto elem, std::array<double, 3> const& value, std::array<double, 3> const&) {
                if (ddc::DiscreteElement<DDimX>(elem).uid() == 0
                    && ddc::DiscreteElement<DDimY>(elem).uid() == 0) {
                    induction = value;
                }
            });

    EXPECT_NEAR(induction[0], 2.0, 1e-12);
    EXPECT_NEAR(induction[1], -1.0, 1e-12);
    EXPECT_DOUBLE_EQ(induction[2], 0.0);
}

TEST(OnelabInterface, MagnetostaticsInductionMomentLocationOnQuadraticField)
{
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetic_induction_moment_from_potential_z;
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;

    std::vector<double> const x_coords {0.0, 1.0, 2.0};
    std::vector<double> const y_coords {0.0, 1.0, 2.0};
    auto node_value_z = [&](std::size_t i, std::size_t j) {
        return x_coords[i] * x_coords[i] + 2.0 * y_coords[j] * y_coords[j];
    };

    auto const elem00 = ddc::DiscreteElement<DDimX, DDimY>(0, 0);
    EXPECT_NEAR(magnetic_induction_moment_from_potential_z<0>(elem00, node_value_z), 2.0, 1e-12);
    EXPECT_NEAR(magnetic_induction_moment_from_potential_z<1>(elem00, node_value_z), -1.0, 1e-12);

    auto const elem10 = ddc::DiscreteElement<DDimX, DDimY>(1, 0);
    EXPECT_NEAR(magnetic_induction_moment_from_potential_z<1>(elem10, node_value_z), -3.0, 1e-12);

    auto const elem01 = ddc::DiscreteElement<DDimX, DDimY>(0, 1);
    EXPECT_NEAR(magnetic_induction_moment_from_potential_z<0>(elem01, node_value_z), 6.0, 1e-12);
}

TEST(OnelabInterface, MagnetostaticsPostProcessForceDensityUsesLibraryCodifferential)
{
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::CellPostProcessFields;
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::
            fill_force_density_on_quadrilateral_grid;

    sil::onelab_interface::gmsh::StructuredGrid2D grid;
    grid.x_coords = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    grid.y_coords = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    grid.z_value = 0.0;

    std::vector<CellPostProcessFields> cell_outputs(grid.ncell_x() * grid.ncell_y());
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            double const x = grid.cell_center_x(i);
            double const y = grid.cell_center_y(j);
            cell_outputs[grid.cell_index(i, j)].maxwell_stress = {
                    x,
                    2.0 * y,
                    0.0,
                    0.0,
                    3.0 * x,
                    4.0 * y,
            };
        }
    }

    fill_force_density_on_quadrilateral_grid(grid, cell_outputs);

    for (std::size_t j = 1; j + 1 < grid.ncell_y(); ++j) {
        for (std::size_t i = 1; i + 1 < grid.ncell_x(); ++i) {
            std::array<double, 3> const& force_density
                    = cell_outputs[grid.cell_index(i, j)].force_density;
            EXPECT_NEAR(force_density[0], 1.0, 1e-12);
            EXPECT_NEAR(force_density[1], 2.0, 1e-12);
            EXPECT_NEAR(force_density[2], 7.0, 1e-12);
        }
    }
}

TEST(OnelabInterface, MagnetostaticsResultViewExportsDirectPostProcessedFields)
{
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::CellInputFields;
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::CellPostProcessFields;
    using similie::onelab_interface::linear_magnetostatics_onelab::detail::write_results_view;

    sil::onelab_interface::gmsh::StructuredGrid2D grid;
    grid.x_coords = {0.0, 1.0};
    grid.y_coords = {0.0, 1.0};
    grid.z_value = 0.0;

    std::vector<CellInputFields> cell_inputs(1);
    cell_inputs[0].mu = 2.0;
    cell_inputs[0].current_density = {0.0, 0.0, 3.0};

    std::vector<CellPostProcessFields> cell_outputs(1);
    cell_outputs[0].magnetic_induction = {4.0, 5.0, 6.0};
    cell_outputs[0].magnetic_field = {7.0, 8.0, 9.0};
    cell_outputs[0].maxwell_stress = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    cell_outputs[0].force_density = {16.0, 17.0, 18.0};

    std::vector<double> magnetic_vector_potential = {
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            4.0,
    };

    std::filesystem::path const output_file
            = std::filesystem::temp_directory_path() / "similie_onelab_result_view_test.pos";
    write_results_view(output_file, grid, cell_inputs, cell_outputs, magnetic_vector_potential);

    std::ifstream input(output_file);
    ASSERT_TRUE(input.is_open());
    std::string const
            contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    EXPECT_NE(
            contents.find("View \"SimiLie linear magnetostatics magnetic induction\""),
            std::string::npos);
    EXPECT_NE(
            contents.find("View \"SimiLie linear magnetostatics magnetic field\""),
            std::string::npos);
    EXPECT_NE(
            contents.find("View \"SimiLie linear magnetostatics Maxwell stress xx\""),
            std::string::npos);
    EXPECT_NE(
            contents.find("View \"SimiLie linear magnetostatics force density\""),
            std::string::npos);
    EXPECT_NE(contents.find("VP(0.5,0.5,0){4,5,6};"), std::string::npos);
    EXPECT_NE(contents.find("VP(0.5,0.5,0){7,8,9};"), std::string::npos);
    EXPECT_NE(contents.find("SP(0.5,0.5,0){13};"), std::string::npos);
    EXPECT_NE(contents.find("VP(0.5,0.5,0){16,17,18};"), std::string::npos);

    std::filesystem::remove(output_file);
}

TEST(OnelabInterface, MagnetostaticsLocalOperatorMatchesItsAssembledMatrix)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using namespace similie::physics::magnetostatics;

    Kokkos::View<double*> x_coords("x", 5);
    Kokkos::View<double*> y_coords("y", 5);
    auto x_host = Kokkos::create_mirror_view(x_coords);
    auto y_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < 5; ++i) {
        x_host(i) = static_cast<double>(i);
        y_host(i) = static_cast<double>(i);
    }
    Kokkos::deep_copy(x_coords, x_host);
    Kokkos::deep_copy(y_coords, y_host);

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>(
                    ddc::DiscreteDomain<DDimX, DDimY>(
                            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                            ddc::DiscreteVector<DDimX, DDimY>(5, 5)),
                    sil::tensor::TensorAccessor<ScalarPotentialIndex>().domain()),
            ddc::KokkosAllocator<double, memory_space>());
    ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            mu_host(i, j, 0) = 1.0;
        }
    }
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);

    auto const hamiltonian = LinearMagnetostaticsHamiltonian(mu_tensor);
    auto const equations = similie::physics::HamiltonEquations {hamiltonian};
    {
        MagnetostaticsOperator2D<memory_space, decltype(equations)> operator_model(
                equations,
                x_coords,
                y_coords,
                similie::solvers::Criterion::MomentsTemporalDerivative);
        {
            auto matrix_data = assemble_matrix_data(operator_model);

            Kokkos::View<double**> input("input", 25, 1);
            Kokkos::View<double**> output("output", 25, 1);
            auto input_host = Kokkos::create_mirror_view(input);
            for (std::size_t j = 0; j < 5; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    input_host(i + 5 * j, 0) = static_cast<double>(i * i + 2 * j);
                }
            }
            Kokkos::deep_copy(input, input_host);

            operator_model.apply(Kokkos::DefaultExecutionSpace(), input, output);
            auto output_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);

            std::vector<double> matrix_output(25, 0.0);
            std::vector<std::vector<double>> dense_matrix(25, std::vector<double>(25, 0.0));
            for (auto const& nz : matrix_data.nonzeros) {
                ASSERT_LT(static_cast<std::size_t>(nz.row), 25U);
                ASSERT_LT(static_cast<std::size_t>(nz.column), 25U);
                matrix_output[static_cast<std::size_t>(nz.row)]
                        += nz.value * input_host(static_cast<std::size_t>(nz.column), 0);
                dense_matrix[static_cast<std::size_t>(nz.row)][static_cast<std::size_t>(nz.column)]
                        += nz.value;
            }

            for (std::size_t row = 0; row < 25; ++row) {
                EXPECT_NEAR(output_host(row, 0), matrix_output[row], 1e-12) << "row=" << row;
            }

            for (std::size_t j = 0; j < 5; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    std::size_t const row = i + 5 * j;
                    if (i == 0 || j == 0 || i + 1 == 5 || j + 1 == 5) {
                        EXPECT_DOUBLE_EQ(dense_matrix[row][row], 1.0);
                        continue;
                    }
                    EXPECT_GT(dense_matrix[row][row], 0.0) << "row=" << row;
                }
            }

            for (std::size_t row = 0; row < 25; ++row) {
                for (std::size_t column = 0; column < 25; ++column) {
                    std::size_t const row_i = row % 5;
                    std::size_t const row_j = row / 5;
                    std::size_t const column_i = column % 5;
                    std::size_t const column_j = column / 5;
                    bool const row_boundary
                            = (row_i == 0 || row_j == 0 || row_i + 1 == 5 || row_j + 1 == 5);
                    bool const column_boundary
                            = (column_i == 0 || column_j == 0 || column_i + 1 == 5
                               || column_j + 1 == 5);
                    if (row_boundary || column_boundary) {
                        continue;
                    }
                    EXPECT_NEAR(dense_matrix[row][column], dense_matrix[column][row], 1e-12)
                            << "row=" << row << " col=" << column;
                }
            }
        }
    }
}

TEST(OnelabInterface, MagnetostaticsMatrixFreeSolverAcceptsDefaultKokkosLayout)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using namespace similie::physics::magnetostatics;

    Kokkos::View<double*> x_coords("x", 5);
    Kokkos::View<double*> y_coords("y", 5);
    auto x_host = Kokkos::create_mirror_view(x_coords);
    auto y_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < 5; ++i) {
        x_host(i) = static_cast<double>(i);
        y_host(i) = static_cast<double>(i);
    }
    Kokkos::deep_copy(x_coords, x_host);
    Kokkos::deep_copy(y_coords, y_host);

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>(
                    ddc::DiscreteDomain<DDimX, DDimY>(
                            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                            ddc::DiscreteVector<DDimX, DDimY>(5, 5)),
                    sil::tensor::TensorAccessor<ScalarPotentialIndex>().domain()),
            ddc::KokkosAllocator<double, memory_space>());
    ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            mu_host(i, j, 0) = 1.0;
        }
    }
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);

    auto const hamiltonian = LinearMagnetostaticsHamiltonian(mu_tensor);
    auto const equations = similie::physics::HamiltonEquations {hamiltonian};
    MagnetostaticsOperator2D<memory_space, decltype(equations)> operator_model(
            equations,
            x_coords,
            y_coords,
            similie::solvers::Criterion::MomentsTemporalDerivative);

    Kokkos::View<double**> rhs("rhs", 25, 1);
    Kokkos::View<double**> solution("solution", 25, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            std::size_t const row = i + 5 * j;
            rhs_host(row, 0) = (i == 0 || j == 0 || i + 1 == 5 || j + 1 == 5) ? 0.0 : 1.0;
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);

    similie::solvers::StrongFormulationSolverSettings settings;
    settings.max_iterations = 200;
    settings.relative_tolerance = 1.0e-10;
    settings.jacobi_max_block_size = 1;
    settings.use_matrix_free = true;

    auto const diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            solution,
            settings);

    EXPECT_GT(diagnostics.iterations, 0U);
    EXPECT_TRUE(std::isfinite(diagnostics.final_residual_l2));
    EXPECT_TRUE(std::isfinite(diagnostics.final_relative_residual));
    EXPECT_TRUE(diagnostics.converged);
}

TEST(OnelabInterface, MagnetostaticsMatrixFreeAndAssembledSolvesMatch)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using namespace similie::physics::magnetostatics;

    Kokkos::View<double*> x_coords("x", 5);
    Kokkos::View<double*> y_coords("y", 5);
    auto x_host = Kokkos::create_mirror_view(x_coords);
    auto y_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < 5; ++i) {
        x_host(i) = static_cast<double>(i);
        y_host(i) = static_cast<double>(i);
    }
    Kokkos::deep_copy(x_coords, x_host);
    Kokkos::deep_copy(y_coords, y_host);

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>(
                    ddc::DiscreteDomain<DDimX, DDimY>(
                            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                            ddc::DiscreteVector<DDimX, DDimY>(5, 5)),
                    sil::tensor::TensorAccessor<ScalarPotentialIndex>().domain()),
            ddc::KokkosAllocator<double, memory_space>());
    ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            mu_host(i, j, 0) = 2.0;
        }
    }
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);

    auto const hamiltonian = LinearMagnetostaticsHamiltonian(mu_tensor);
    auto const equations = similie::physics::HamiltonEquations {hamiltonian};
    MagnetostaticsOperator2D<memory_space, decltype(equations)> operator_model(
            equations,
            x_coords,
            y_coords,
            similie::solvers::Criterion::MomentsTemporalDerivative);

    Kokkos::View<double**> rhs("rhs", 25, 1);
    Kokkos::View<double**> solution_matrix_free("solution_matrix_free", 25, 1);
    Kokkos::View<double**> solution_assembled("solution_assembled", 25, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            std::size_t const row = i + 5 * j;
            rhs_host(row, 0) = (i == 0 || j == 0 || i + 1 == 5 || j + 1 == 5)
                                       ? 0.0
                                       : static_cast<double>(1 + i + 2 * j);
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);

    similie::solvers::StrongFormulationSolverSettings matrix_free_settings;
    matrix_free_settings.max_iterations = 200;
    matrix_free_settings.relative_tolerance = 1.0e-12;
    matrix_free_settings.jacobi_max_block_size = 1;
    matrix_free_settings.use_matrix_free = true;

    similie::solvers::StrongFormulationSolverSettings assembled_settings = matrix_free_settings;
    assembled_settings.use_matrix_free = false;

    auto const matrix_free_diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            solution_matrix_free,
            matrix_free_settings);
    auto const assembled_diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            solution_assembled,
            assembled_settings);

    auto matrix_free_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            solution_matrix_free);
    auto assembled_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_assembled);

    EXPECT_TRUE(matrix_free_diagnostics.converged);
    EXPECT_TRUE(assembled_diagnostics.converged);
    for (std::size_t row = 0; row < 25; ++row) {
        EXPECT_NEAR(matrix_free_host(row, 0), assembled_host(row, 0), 1e-10) << "row=" << row;
    }
}

TEST(OnelabInterface, MagnetostaticsMatrixFreeAndAssembledSolvesMatchOnNonuniformVariableMu)
{
    using namespace similie::onelab_interface::linear_magnetostatics_onelab::detail::
            magnetostatics_local;
    using namespace similie::physics::magnetostatics;

    Kokkos::View<double*> x_coords("x", 7);
    Kokkos::View<double*> y_coords("y", 6);
    auto x_host = Kokkos::create_mirror_view(x_coords);
    auto y_host = Kokkos::create_mirror_view(y_coords);
    double const x_values[7] = {0.0, 0.3, 0.9, 1.0, 1.8, 2.7, 4.0};
    double const y_values[6] = {0.0, 0.2, 0.6, 1.5, 1.7, 3.0};
    for (std::size_t i = 0; i < 7; ++i) {
        x_host(i) = x_values[i];
    }
    for (std::size_t j = 0; j < 6; ++j) {
        y_host(j) = y_values[j];
    }
    Kokkos::deep_copy(x_coords, x_host);
    Kokkos::deep_copy(y_coords, y_host);

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>(
                    ddc::DiscreteDomain<DDimX, DDimY>(
                            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                            ddc::DiscreteVector<DDimX, DDimY>(7, 6)),
                    sil::tensor::TensorAccessor<ScalarPotentialIndex>().domain()),
            ddc::KokkosAllocator<double, memory_space>());
    ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    for (std::size_t j = 0; j < 6; ++j) {
        for (std::size_t i = 0; i < 7; ++i) {
            mu_host(i, j, 0) = 1.0 + 0.1 * static_cast<double>(i) + 0.2 * static_cast<double>(j);
        }
    }
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);

    auto const hamiltonian = LinearMagnetostaticsHamiltonian(mu_tensor);
    auto const equations = similie::physics::HamiltonEquations {hamiltonian};
    MagnetostaticsOperator2D<memory_space, decltype(equations)> operator_model(
            equations,
            x_coords,
            y_coords,
            similie::solvers::Criterion::MomentsTemporalDerivative);

    Kokkos::View<double**> rhs("rhs", 42, 1);
    Kokkos::View<double**> solution_matrix_free("solution_matrix_free", 42, 1);
    Kokkos::View<double**> solution_assembled("solution_assembled", 42, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < 6; ++j) {
        for (std::size_t i = 0; i < 7; ++i) {
            std::size_t const row = i + 7 * j;
            rhs_host(row, 0) = (i == 0 || j == 0 || i + 1 == 7 || j + 1 == 6)
                                       ? 0.0
                                       : static_cast<double>(1 + 3 * i - 2 * j);
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);

    similie::solvers::StrongFormulationSolverSettings matrix_free_settings;
    matrix_free_settings.max_iterations = 400;
    matrix_free_settings.relative_tolerance = 1.0e-12;
    matrix_free_settings.jacobi_max_block_size = 1;
    matrix_free_settings.use_matrix_free = true;

    similie::solvers::StrongFormulationSolverSettings assembled_settings = matrix_free_settings;
    assembled_settings.use_matrix_free = false;

    auto const matrix_free_diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            solution_matrix_free,
            matrix_free_settings);
    auto const assembled_diagnostics = similie::solvers::minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace(),
            operator_model,
            rhs,
            solution_assembled,
            assembled_settings);

    auto matrix_free_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            solution_matrix_free);
    auto assembled_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_assembled);

    EXPECT_TRUE(matrix_free_diagnostics.converged);
    EXPECT_TRUE(assembled_diagnostics.converged);
    for (std::size_t row = 0; row < 42; ++row) {
        double const matrix_free_value = matrix_free_host(row, 0);
        double const assembled_value = assembled_host(row, 0);
        double const scale = std::max({1.0, std::abs(matrix_free_value), std::abs(assembled_value)});
        EXPECT_LE(std::abs(matrix_free_value - assembled_value) / scale, 1e-12) << "row=" << row;
    }
}
