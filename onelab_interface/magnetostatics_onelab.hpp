// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <ginkgo/core/base/matrix_data.hpp>
#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/codifferential.hpp>
#include <similie/exterior/hodge_star.hpp>
#include <similie/exterior/reduction_and_reconstruction.hpp>
#include <similie/misc/macros.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics_constitutive_law.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/magnetostatics/nonlinear_magnetostatics.hpp>
#include <similie/solvers/minimize_strong_formulation_residual.hpp>

#include <Kokkos_Core.hpp>

#include "gmsh_structured_grid.hpp"

namespace similie::onelab_interface::magnetostatics_onelab {

struct Inputs
{
    double current_density_magnitude = 0.0;
    double num_turns = 1.0;
    double core_mu = 0.0;
    double mu0 = 0.0;
    double length_z = 1.0;
    std::vector<int> positive_electrical_conductor_tags;
    std::vector<int> negative_electrical_conductor_tags;
    std::vector<int> magnetic_material_tags;
    std::vector<int> diagnostic_region_tags;
    bool use_nonlinear_magnetic_material = false;
    std::string nonlinear_bh_curve = "EIcore";
    std::vector<double> nonlinear_b_samples;
    std::vector<double> nonlinear_h_samples;
};

struct Result
{
    std::string topology;
    std::size_t node_count = 0;
    std::array<std::size_t, 3> mesh_dimensions {0, 0, 0};
    std::size_t num_cells = 0;
    std::size_t num_air_cells = 0;
    std::size_t num_core_cells = 0;
    std::size_t num_coil_cells = 0;
    std::size_t num_diagnostic_faces = 0;
    std::size_t num_current_cells = 0;
    double max_abs_potential = 0.0;
    double max_abs_induction = 0.0;
    double max_abs_field = 0.0;
    double diagnostic_surface_measure = 0.0;
    double diagnostic_traction_magnitude_integral = 0.0;
    double diagnostic_flux_integral = 0.0;
    double diagnostic_current_integral = 0.0;
    std::array<double, 3> diagnostic_traction_integral {0.0, 0.0, 0.0};
    solvers::StrongFormulationSolverDiagnostics solver_diagnostics;
};

template <class Problem, class ProblemParameterName, class PublishString, class PublishNumber>
void synchronize_controls(
        Problem const& problem,
        ProblemParameterName&& problem_parameter_name,
        PublishString&& publish_or_sync_string,
        PublishNumber&& publish_or_sync_number)
{
    publish_or_sync_string(
            problem_parameter_name("2LinearMagnetostatics", "0Preprocess"),
            "Preprocess",
            "Linear magnetostatics preprocessing strategy selected in the .silpro file.",
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .name,
            true);
    publish_or_sync_string(
            problem_parameter_name("2LinearMagnetostatics", "1PostProcess"),
            "PostProcess",
            "Linear magnetostatics postprocess strategy selected in the .silpro file.",
            problem.force_density_diagnostics_postprocess.name,
            true);
}

template <class Problem, class ReadNumberParameter, class ReadRequiredIntegerParameter>
Inputs read_inputs(
        Problem const& problem,
        ReadNumberParameter&& read_number_parameter,
        ReadRequiredIntegerParameter&& read_required_integer_parameter)
{
    double const current_density_magnitude = read_number_parameter(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .current_density_z_parameter,
            std::nullopt,
            std::numeric_limits<double>::quiet_NaN());
    double const core_mu = read_number_parameter(
            problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                    .magnetic_permeability_parameter,
            std::nullopt,
            std::numeric_limits<double>::quiet_NaN());
    double const num_turns = read_number_parameter(
            "Input/4Coil Parameters/1Number of turns",
            std::nullopt,
            std::numeric_limits<double>::quiet_NaN());
    double const length_z = read_number_parameter(
            "Input/10Geometric dimensions/00Length along z-axis [m]",
            std::nullopt,
            1.0);
    double const mu0 = 4.e-7 * std::numbers::pi_v<double>;

    if (!(current_density_magnitude > 0.0)) {
        throw std::runtime_error(
                "missing or invalid 'Input/90SimiLie/0Coil current density magnitude z [A/m^2]' "
                "ONELAB parameter");
    }
    if (!(num_turns > 0.0)) {
        throw std::runtime_error(
                "missing or invalid 'Input/4Coil Parameters/1Number of turns' ONELAB parameter");
    }
    if (!problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                 .use_nonlinear_magnetic_material
        && !(core_mu > 0.0)) {
        throw std::runtime_error(
                "missing or invalid 'Input/90SimiLie/1Core magnetic permeability [H/m]' ONELAB "
                "parameter");
    }

    Inputs inputs;
    inputs.current_density_magnitude = current_density_magnitude;
    inputs.num_turns = num_turns;
    inputs.core_mu = core_mu > 0.0 ? core_mu : mu0;
    inputs.mu0 = mu0;
    inputs.length_z = length_z > 0.0 ? length_z : 1.0;
    inputs.use_nonlinear_magnetic_material
            = problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                      .use_nonlinear_magnetic_material;
    inputs.nonlinear_bh_curve
            = problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                      .nonlinear_bh_curve;
    for (std::string const& parameter_name :
         problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                 .positive_electrical_conductor_tags) {
        inputs.positive_electrical_conductor_tags.push_back(
                read_required_integer_parameter(parameter_name));
    }
    for (std::string const& parameter_name :
         problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                 .negative_electrical_conductor_tags) {
        inputs.negative_electrical_conductor_tags.push_back(
                read_required_integer_parameter(parameter_name));
    }
    for (std::string const& parameter_name :
         problem.single_electrical_conductor_material_with_single_linear_magnetic_material_preprocess
                 .magnetic_material_tags) {
        inputs.magnetic_material_tags.push_back(read_required_integer_parameter(parameter_name));
    }
    for (std::string const& parameter_name :
         problem.force_density_diagnostics_postprocess.diagnostic_region_tags) {
        inputs.diagnostic_region_tags.push_back(read_required_integer_parameter(parameter_name));
    }
    return inputs;
}

template <class PublishOutputString, class PublishOutputNumber, class PublishStatus>
void publish_outputs(
        std::filesystem::path const& mesh_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        Result const& result,
        PublishOutputString&& publish_output_string,
        PublishOutputNumber&& publish_output_number,
        PublishStatus&& publish_status)
{
    publish_output_string(
            "Mesh file",
            mesh_file.string(),
            "Mesh file",
            "Mesh file exported by Gmsh for the linear magnetostatics interface.",
            "file");
    publish_output_number(
            "Coil current density magnitude [A/m^2]",
            inputs.current_density_magnitude,
            "Coil current density magnitude [A/m^2]",
            "Current density magnitude read from the ONELAB model inputs.");
    publish_output_number(
            "Number of turns",
            inputs.num_turns,
            "Number of turns",
            "Number of conductor turns used by the homogenized winding source.");
    publish_output_number(
            "Air permeability [H/m]",
            inputs.mu0,
            "Air permeability [H/m]",
            "Magnetic permeability used in air and coil cells.");
    publish_output_number(
            "Core permeability [H/m]",
            inputs.core_mu,
            "Core permeability [H/m]",
            inputs.use_nonlinear_magnetic_material ? "Reference linear permeability used outside "
                                                     "the nonlinear constitutive update."
                                                   : "Magnetic permeability read from the ONELAB "
                                                     "model inputs and used in core cells.");
    publish_output_string(
            "Core constitutive law",
            inputs.use_nonlinear_magnetic_material ? inputs.nonlinear_bh_curve
                                                   : "LinearPermeability",
            "Core constitutive law",
            "Core constitutive law selected for the magnetostatics solve.",
            "generic");
    publish_output_number(
            "Number of air cells",
            static_cast<double>(result.num_air_cells),
            "Number of air cells",
            "Number of " + result.topology + " cells tagged as air.");
    publish_output_number(
            "Number of core cells",
            static_cast<double>(result.num_core_cells),
            "Number of core cells",
            "Number of " + result.topology + " cells tagged as linear magnetic material.");
    publish_output_number(
            "Number of coil cells",
            static_cast<double>(result.num_coil_cells),
            "Number of coil cells",
            "Number of " + result.topology + " cells tagged as conductor.");
    publish_output_number(
            "Solver iterations",
            static_cast<double>(result.solver_diagnostics.iterations),
            "Solver iterations",
            "Number of conjugate-gradient iterations performed by the stationary "
            "strong-formulation solver.");
    publish_output_string(
            "Solver backend",
            solver_settings.use_matrix_free ? "matrix-free" : "assembled-matrix",
            "Solver backend",
            "Backend used by the stationary strong-formulation solver.",
            "generic");
    publish_output_number(
            "Solver converged",
            result.solver_diagnostics.converged ? 1.0 : 0.0,
            "Solver converged",
            "Equals 1 when the stationary strong-formulation solver met its relative-residual "
            "target, 0 otherwise.");
    publish_output_number(
            "Final residual L2",
            result.solver_diagnostics.final_residual_l2,
            "Final residual L2",
            "Final L2 norm of the strong-formulation residual returned by the stationary solver.");
    publish_output_number(
            "Final relative residual",
            result.solver_diagnostics.final_relative_residual,
            "Final relative residual",
            "Final residual divided by the initial residual, as returned by the stationary "
            "solver.");
    publish_output_number(
            "Solver duration [s]",
            result.solver_diagnostics.duration,
            "Solver duration [s]",
            "Wall-clock time spent inside the iterative solver, excluding matrix and "
            "preconditioner assembly.");
    publish_output_number(
            "Maximum magnetic vector potential [SI]",
            result.max_abs_potential,
            "Maximum magnetic vector potential [SI]",
            "Maximum absolute value of the computed magnetic vector potential.");
    publish_output_number(
            "Maximum magnetic induction [T]",
            result.max_abs_induction,
            "Maximum magnetic induction [T]",
            "Maximum absolute value of the computed magnetic induction.");
    publish_output_number(
            "Maximum magnetic field [A/m]",
            result.max_abs_field,
            "Maximum magnetic field [A/m]",
            "Maximum absolute value of the computed magnetic field.");
    publish_output_number(
            "Upper air-gap flux integral [Wb]",
            result.diagnostic_flux_integral,
            "Upper air-gap flux integral [Wb]",
            "Integral of the normal magnetic induction over the detected upper air-gap surface.");
    publish_output_number(
            "Positive-conductor current integral [A]",
            result.diagnostic_current_integral,
            "Positive-conductor current integral [A]",
            "Integral of the homogenized current density over the positively oriented conductor "
            "region.");
    publish_output_number(
            "Positive-conductor terminal current [A]",
            result.diagnostic_current_integral / inputs.num_turns,
            "Positive-conductor terminal current [A]",
            "Positive-conductor current integral divided by the number of turns.");
    publish_output_number(
            "Inductance from upper air-gap flux [H]",
            result.diagnostic_current_integral == 0.0
                    ? 0.0
                    : result.diagnostic_flux_integral
                              / (result.diagnostic_current_integral * inputs.num_turns),
            "Inductance from upper air-gap flux [H]",
            "Inductance estimated as L = Phi / (N I) with Phi the upper air-gap flux integral, "
            "I the numerically integrated positive-conductor current, and N the number of turns.");
    std::string const integrated_traction_unit = "N";
    publish_output_number(
            "Number of upper air-gap faces",
            static_cast<double>(result.num_diagnostic_faces),
            "Number of upper air-gap faces",
            "Number of air-gap faces detected on the upper boundary of the configured "
            "diagnostic region.");
    publish_output_number(
            "Upper air-gap surface measure",
            result.diagnostic_surface_measure,
            "Upper air-gap surface measure",
            "Effective surface area of the detected upper air-gap surface. In 2D this uses the "
            "configured extrusion length along z.");
    publish_output_number(
            "Integrated upper air-gap traction x [" + integrated_traction_unit + "]",
            result.diagnostic_traction_integral[0],
            "Integrated upper air-gap traction x [" + integrated_traction_unit + "]",
            "Integral of the Maxwell traction T n over the upper air-gap surface, x component.");
    publish_output_number(
            "Integrated upper air-gap traction y [" + integrated_traction_unit + "]",
            result.diagnostic_traction_integral[1],
            "Integrated upper air-gap traction y [" + integrated_traction_unit + "]",
            "Integral of the Maxwell traction T n over the upper air-gap surface, y component.");
    publish_output_number(
            "Integrated upper air-gap traction z [" + integrated_traction_unit + "]",
            result.diagnostic_traction_integral[2],
            "Integrated upper air-gap traction z [" + integrated_traction_unit + "]",
            "Integral of the Maxwell traction T n over the upper air-gap surface, z component.");
    publish_output_number(
            "Mean upper air-gap traction magnitude [Pa]",
            result.diagnostic_surface_measure == 0.0 ? 0.0
                                                     : result.diagnostic_traction_magnitude_integral
                                                               / result.diagnostic_surface_measure,
            "Mean upper air-gap traction magnitude [Pa]",
            "Mean magnitude of the Maxwell traction T n over the detected upper air-gap "
            "surface.");
    publish_status(
            inputs.use_nonlinear_magnetic_material ? "Nonlinear magnetostatics solve completed"
                                                   : "Linear magnetostatics solve completed");
}

namespace detail {

struct CellInputFields
{
    double mu = 0.0;
    std::array<double, 3> current_density {0.0, 0.0, 0.0};
    bool nonlinear_material = false;
};

struct CellPostProcessFields
{
    std::array<double, 3> magnetic_induction {0.0, 0.0, 0.0};
    std::array<double, 3> magnetic_field {0.0, 0.0, 0.0};
    std::array<double, 6> maxwell_stress {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 3> force_density {0.0, 0.0, 0.0};
};

struct DiagnosticFaceSample
{
    double x = 0.0;
    double z = 0.0;
    double measure = 0.0;
    std::array<double, 3> traction {0.0, 0.0, 0.0};
};

template <class Logger>
void log_info(Logger&& logger, std::string const& message)
{
    if constexpr (std::is_invocable_v<Logger, std::string const&>) {
        logger(message);
    }
}

inline bool has_tag(std::vector<int> const& tags, int physical_tag)
{
    return std::find(tags.begin(), tags.end(), physical_tag) != tags.end();
}

[[nodiscard]] inline std::array<double, 3> traction_on_positive_y_face(
        std::array<double, 3> const& magnetic_induction,
        std::array<double, 3> const& magnetic_field)
{
    double const half_trace = 0.5
                              * (magnetic_induction[0] * magnetic_field[0]
                                 + magnetic_induction[1] * magnetic_field[1]
                                 + magnetic_induction[2] * magnetic_field[2]);
    return {
            magnetic_induction[0] * magnetic_field[1],
            magnetic_induction[1] * magnetic_field[1] - half_trace,
            magnetic_induction[2] * magnetic_field[1],
    };
}

inline void apply_x_mirror_symmetry_projection_to_traction_integral(
        std::vector<DiagnosticFaceSample> const& face_samples,
        Result& result)
{
    if (face_samples.empty()) {
        return;
    }

    constexpr double tol = 1.0e-12;
    std::vector<DiagnosticFaceSample> sorted_samples(face_samples);
    std::
            sort(sorted_samples.begin(),
                 sorted_samples.end(),
                 [](DiagnosticFaceSample const& lhs, DiagnosticFaceSample const& rhs) {
                     if (std::abs(lhs.z - rhs.z) > tol) {
                         return lhs.z < rhs.z;
                     }
                     return lhs.x < rhs.x;
                 });

    std::array<double, 3> projected_integral {0.0, 0.0, 0.0};
    std::size_t left = 0;
    std::size_t right = sorted_samples.size();
    while (left < right) {
        --right;
        DiagnosticFaceSample const& left_face = sorted_samples[left];
        DiagnosticFaceSample const& right_face = sorted_samples[right];
        if (left == right) {
            if (std::abs(left_face.x) > tol) {
                return;
            }
            projected_integral[1] += left_face.measure * left_face.traction[1];
            projected_integral[2] += left_face.measure * left_face.traction[2];
            break;
        }

        if (std::abs(left_face.z - right_face.z) > tol || std::abs(left_face.x + right_face.x) > tol
            || std::abs(left_face.measure - right_face.measure) > tol) {
            return;
        }

        projected_integral[1]
                += left_face.measure * (left_face.traction[1] + right_face.traction[1]);
        projected_integral[2]
                += left_face.measure * (left_face.traction[2] + right_face.traction[2]);
        ++left;
    }

    result.diagnostic_traction_integral = projected_integral;
}

inline void validate_nonlinear_bh_curve(std::string const& nonlinear_bh_curve)
{
    if (nonlinear_bh_curve != "EIcore") {
        throw std::runtime_error(
                "unsupported nonlinear magnetostatics B-H curve '" + nonlinear_bh_curve
                + "' (supported: EIcore)");
    }
}

inline std::vector<double> parse_number_list(std::string const& values)
{
    std::vector<double> parsed;
    std::string token;
    for (char const c : values) {
        if (c == ',' || std::isspace(static_cast<unsigned char>(c)) != 0) {
            if (!token.empty()) {
                parsed.push_back(std::stod(token));
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) {
        parsed.push_back(std::stod(token));
    }
    return parsed;
}

inline std::vector<double> read_bh_curve_component(
        std::string const& bh_pro_content,
        std::string const& curve_name,
        char component_name)
{
    std::string const signature = "Mat" + curve_name + "_" + component_name + "()";
    std::size_t const signature_pos = bh_pro_content.find(signature);
    if (signature_pos == std::string::npos) {
        throw std::runtime_error(
                "could not find B-H curve component '" + signature + "' in BH.pro");
    }
    std::size_t const open_brace = bh_pro_content.find('{', signature_pos);
    std::size_t const close_brace = bh_pro_content.find('}', open_brace);
    if (open_brace == std::string::npos || close_brace == std::string::npos) {
        throw std::runtime_error("malformed B-H curve component '" + signature + "' in BH.pro");
    }
    return parse_number_list(bh_pro_content.substr(open_brace + 1, close_brace - open_brace - 1));
}

inline void load_bh_curve_from_bh_pro(Inputs& inputs, std::filesystem::path const& bh_pro_file)
{
    std::ifstream stream(bh_pro_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open nonlinear B-H file: " + bh_pro_file.string());
    }
    std::string const
            content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    inputs.nonlinear_h_samples = read_bh_curve_component(content, inputs.nonlinear_bh_curve, 'h');
    inputs.nonlinear_b_samples = read_bh_curve_component(content, inputs.nonlinear_bh_curve, 'b');
    if (inputs.nonlinear_h_samples.size() != inputs.nonlinear_b_samples.size()
        || inputs.nonlinear_b_samples.size() < 2) {
        throw std::runtime_error(
                "invalid nonlinear B-H curve samples parsed from " + bh_pro_file.string());
    }
}

template <class MemorySpace>
using coord_view_type = Kokkos::View<double const*, MemorySpace>;

namespace magnetostatics_local {

struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

struct Z
{
    static constexpr bool PERIODIC = false;
};

template <class MuTensor>
using LinearMagnetostaticsHamiltonian
        = physics::magnetostatics::LinearMagnetostaticsHamiltonian<MuTensor, X, Y, Z>;
using MagneticVectorPotentialToMagneticInduction
        = physics::magnetostatics::MagneticVectorPotentialToMagneticInduction<X, Y, Z>;
using InPlaneIndex = sil::tensor::TensorNaturalIndex<X, Y>;

struct MagneticMoments
{
    double x;
    double y;
    double z;

    template <class Index>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double get() const
    {
        if constexpr (std::is_same_v<Index, X>) {
            return x;
        } else if constexpr (std::is_same_v<Index, Y>) {
            return y;
        } else if constexpr (std::is_same_v<Index, Z>) {
            return z;
        } else {
            static_assert(
                    std::is_same_v<Index, X> || std::is_same_v<Index, Y>
                            || std::is_same_v<Index, Z>,
                    "unsupported magnetostatics moment tag");
        }
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double norm2() const
    {
        return x * x + y * y + z * z;
    }
};

struct DDimX
{
    using continuous_dimension_type = X;
    static constexpr bool PERIODIC = false;
};

struct DDimY
{
    using continuous_dimension_type = Y;
    static constexpr bool PERIODIC = false;
};

using NodeDomain2D = ddc::DiscreteDomain<DDimX, DDimY>;
using ScalarPotentialIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;

template <class MemorySpace>
using scalar_tensor_alloc_type = ddc::Chunk<
        double,
        ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>,
        ddc::KokkosAllocator<double, MemorySpace>>;

template <class MemorySpace>
using ScalarPotentialTensor2D = sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<DDimX, DDimY, ScalarPotentialIndex>,
        Kokkos::layout_right,
        MemorySpace>;

template <std::size_t NumSamples>
std::array<double, NumSamples> to_std_array(std::vector<double> const& values)
{
    if (values.size() != NumSamples) {
        throw std::runtime_error("unexpected nonlinear B-H curve sample count");
    }
    std::array<double, NumSamples> array {};
    std::copy_n(values.begin(), NumSamples, array.begin());
    return array;
}

template <std::size_t MaxSamples>
std::array<double, MaxSamples> to_padded_std_array(std::vector<double> const& values)
{
    if (values.size() > MaxSamples) {
        throw std::runtime_error("nonlinear B-H curve exceeds supported storage capacity");
    }
    std::array<double, MaxSamples> array {};
    std::copy(values.begin(), values.end(), array.begin());
    return array;
}

template <class LinearPermeabilityTensor, class NonlinearMaskTensor, class Curve>
class MixedMaterialMagnetostaticsEquations
{
    LinearPermeabilityTensor m_linear_permeability;
    NonlinearMaskTensor m_nonlinear_mask;
    Curve m_nonlinear_bh_curve;

public:
    static constexpr bool IS_LINEAR = false;

    MixedMaterialMagnetostaticsEquations(
            LinearPermeabilityTensor linear_permeability,
            NonlinearMaskTensor nonlinear_mask,
            Curve nonlinear_bh_curve)
        : m_linear_permeability(linear_permeability)
        , m_nonlinear_mask(nonlinear_mask)
        , m_nonlinear_bh_curve(nonlinear_bh_curve)
    {
    }

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION bool is_nonlinear(Elem elem) const
    {
        return m_nonlinear_mask(
                       elem,
                       ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>>(0))
               > 0.5;
    }

    template <class Elem>
    [[nodiscard]] KOKKOS_FUNCTION double linear_mu(Elem elem) const
    {
        return m_linear_permeability(
                elem,
                ddc::DiscreteElement<sil::tensor::Covariant<sil::tensor::ScalarIndex>>(0));
    }

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION double dpotential_dt(MagneticMoments moments, Elem elem) const
    {
        if (is_nonlinear(elem)) {
            return m_nonlinear_bh_curve.nu_from_q(moments.norm2()) * moments.template get<Index>();
        }
        return moments.template get<Index>() / linear_mu(elem);
    }

    template <class RowIndex, class ColumnIndex, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION double jacobian(MagneticMoments moments, Elem elem) const
    {
        if (!is_nonlinear(elem)) {
            return std::is_same_v<RowIndex, ColumnIndex> ? 1.0 / linear_mu(elem) : 0.0;
        }
        double const q = moments.norm2();
        double const nu = m_nonlinear_bh_curve.nu_from_q(q);
        double const dnu = m_nonlinear_bh_curve.dnu_dq(q);
        return (std::is_same_v<RowIndex, ColumnIndex> ? nu : 0.0)
               + 2.0 * dnu * moments.template get<RowIndex>() * moments.template get<ColumnIndex>();
    }
};

template <class MemorySpace, class Equations>
class MagnetostaticsOperator2D
{
public:
    using coord_view_type = detail::coord_view_type<MemorySpace>;
    static constexpr bool IS_LINEAR = Equations::IS_LINEAR;
    static constexpr int MOMENT_STENCIL_MAX_SIZE = 4;
    static constexpr int TRANSPOSED_MOMENT_STENCIL_MAX_SIZE = 64;
    static constexpr int OUTER_STENCIL_MAX_SIZE = 4;

private:
    Equations m_equations;
    coord_view_type m_x_coords;
    coord_view_type m_y_coords;
    solvers::Criterion m_criterion;
    std::size_t m_nx;
    std::size_t m_ny;
    NodeDomain2D m_node_domain;
    Kokkos::View<int* [MOMENT_STENCIL_MAX_SIZE], MemorySpace> m_moment0_columns;
    Kokkos::View<double* [MOMENT_STENCIL_MAX_SIZE], MemorySpace> m_moment0_coefficients;
    Kokkos::View<int* [MOMENT_STENCIL_MAX_SIZE], MemorySpace> m_moment1_columns;
    Kokkos::View<double* [MOMENT_STENCIL_MAX_SIZE], MemorySpace> m_moment1_coefficients;
    Kokkos::View<int*, MemorySpace> m_moment0_counts;
    Kokkos::View<int*, MemorySpace> m_moment1_counts;
    Kokkos::View<int**, Kokkos::LayoutRight, MemorySpace> m_transposed_moment0_columns;
    Kokkos::View<double**, Kokkos::LayoutRight, MemorySpace> m_transposed_moment0_coefficients;
    Kokkos::View<int**, Kokkos::LayoutRight, MemorySpace> m_transposed_moment1_columns;
    Kokkos::View<double**, Kokkos::LayoutRight, MemorySpace> m_transposed_moment1_coefficients;
    Kokkos::View<int*, MemorySpace> m_transposed_moment0_counts;
    Kokkos::View<int*, MemorySpace> m_transposed_moment1_counts;
    Kokkos::View<int* [OUTER_STENCIL_MAX_SIZE], MemorySpace> m_outer0_columns;
    Kokkos::View<double* [OUTER_STENCIL_MAX_SIZE], MemorySpace> m_outer0_coefficients;
    Kokkos::View<int* [OUTER_STENCIL_MAX_SIZE], MemorySpace> m_outer1_columns;
    Kokkos::View<double* [OUTER_STENCIL_MAX_SIZE], MemorySpace> m_outer1_coefficients;
    Kokkos::View<int*, MemorySpace> m_outer0_counts;
    Kokkos::View<int*, MemorySpace> m_outer1_counts;

public:
    MagnetostaticsOperator2D(
            Equations equations,
            coord_view_type x_coords,
            coord_view_type y_coords,
            solvers::Criterion criterion)
        : m_equations(std::move(equations))
        , m_x_coords(x_coords)
        , m_y_coords(y_coords)
        , m_criterion(criterion)
        , m_nx(x_coords.extent(0))
        , m_ny(y_coords.extent(0))
        , m_node_domain(
                  ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                  ddc::DiscreteVector<DDimX, DDimY>(m_nx, m_ny))
        , m_moment0_columns("similie_moment0_columns", m_nx * m_ny)
        , m_moment0_coefficients("similie_moment0_coefficients", m_nx * m_ny)
        , m_moment1_columns("similie_moment1_columns", m_nx * m_ny)
        , m_moment1_coefficients("similie_moment1_coefficients", m_nx * m_ny)
        , m_moment0_counts("similie_moment0_counts", m_nx * m_ny)
        , m_moment1_counts("similie_moment1_counts", m_nx * m_ny)
        , m_transposed_moment0_columns(
                  "similie_transposed_moment0_columns",
                  m_nx * m_ny,
                  TRANSPOSED_MOMENT_STENCIL_MAX_SIZE)
        , m_transposed_moment0_coefficients(
                  "similie_transposed_moment0_coefficients",
                  m_nx * m_ny,
                  TRANSPOSED_MOMENT_STENCIL_MAX_SIZE)
        , m_transposed_moment1_columns(
                  "similie_transposed_moment1_columns",
                  m_nx * m_ny,
                  TRANSPOSED_MOMENT_STENCIL_MAX_SIZE)
        , m_transposed_moment1_coefficients(
                  "similie_transposed_moment1_coefficients",
                  m_nx * m_ny,
                  TRANSPOSED_MOMENT_STENCIL_MAX_SIZE)
        , m_transposed_moment0_counts("similie_transposed_moment0_counts", m_nx * m_ny)
        , m_transposed_moment1_counts("similie_transposed_moment1_counts", m_nx * m_ny)
        , m_outer0_columns("similie_outer0_columns", m_nx * m_ny)
        , m_outer0_coefficients("similie_outer0_coefficients", m_nx * m_ny)
        , m_outer1_columns("similie_outer1_columns", m_nx * m_ny)
        , m_outer1_coefficients("similie_outer1_coefficients", m_nx * m_ny)
        , m_outer0_counts("similie_outer0_counts", m_nx * m_ny)
        , m_outer1_counts("similie_outer1_counts", m_nx * m_ny)
    {
        auto moment0_columns_host = Kokkos::create_mirror_view(m_moment0_columns);
        auto moment0_coefficients_host = Kokkos::create_mirror_view(m_moment0_coefficients);
        auto moment1_columns_host = Kokkos::create_mirror_view(m_moment1_columns);
        auto moment1_coefficients_host = Kokkos::create_mirror_view(m_moment1_coefficients);
        auto moment0_counts_host = Kokkos::create_mirror_view(m_moment0_counts);
        auto moment1_counts_host = Kokkos::create_mirror_view(m_moment1_counts);
        auto transposed_moment0_columns_host
                = Kokkos::create_mirror_view(m_transposed_moment0_columns);
        auto transposed_moment0_coefficients_host
                = Kokkos::create_mirror_view(m_transposed_moment0_coefficients);
        auto transposed_moment1_columns_host
                = Kokkos::create_mirror_view(m_transposed_moment1_columns);
        auto transposed_moment1_coefficients_host
                = Kokkos::create_mirror_view(m_transposed_moment1_coefficients);
        auto transposed_moment0_counts_host
                = Kokkos::create_mirror_view(m_transposed_moment0_counts);
        auto transposed_moment1_counts_host
                = Kokkos::create_mirror_view(m_transposed_moment1_counts);
        auto outer0_columns_host = Kokkos::create_mirror_view(m_outer0_columns);
        auto outer0_coefficients_host = Kokkos::create_mirror_view(m_outer0_coefficients);
        auto outer1_columns_host = Kokkos::create_mirror_view(m_outer1_columns);
        auto outer1_coefficients_host = Kokkos::create_mirror_view(m_outer1_coefficients);
        auto outer0_counts_host = Kokkos::create_mirror_view(m_outer0_counts);
        auto outer1_counts_host = Kokkos::create_mirror_view(m_outer1_counts);
        using OutputIndex = sil::exterior::
                coboundary_index_t<sil::tensor::Covariant<InPlaneIndex>, ScalarPotentialIndex>;
        auto const output_y
                = sil::tensor::TensorAccessor<OutputIndex>().template access_element<Y>();
        auto const output_x
                = sil::tensor::TensorAccessor<OutputIndex>().template access_element<X>();
        for (std::size_t j = 0; j < m_ny; ++j) {
            for (std::size_t i = 0; i < m_nx; ++i) {
                std::size_t const row = flat_index(i, j);
                auto const elem = ddc::DiscreteElement<DDimX, DDimY>(i, j);
                for (int slot = 0; slot < MOMENT_STENCIL_MAX_SIZE; ++slot) {
                    moment0_columns_host(row, slot) = 0;
                    moment0_coefficients_host(row, slot) = 0.0;
                    moment1_columns_host(row, slot) = 0;
                    moment1_coefficients_host(row, slot) = 0.0;
                }
                for (int slot = 0; slot < TRANSPOSED_MOMENT_STENCIL_MAX_SIZE; ++slot) {
                    transposed_moment0_columns_host(row, slot) = 0;
                    transposed_moment0_coefficients_host(row, slot) = 0.0;
                    transposed_moment1_columns_host(row, slot) = 0;
                    transposed_moment1_coefficients_host(row, slot) = 0.0;
                }
                for (int slot = 0; slot < OUTER_STENCIL_MAX_SIZE; ++slot) {
                    outer0_columns_host(row, slot) = 0;
                    outer0_coefficients_host(row, slot) = 0.0;
                    outer1_columns_host(row, slot) = 0;
                    outer1_coefficients_host(row, slot) = 0.0;
                }
                int moment0_count = 0;
                auto moment0_stencil
                        = MagneticVectorPotentialToMagneticInduction::template forward_value<X>(
                                elem);
                ddc::device_for_each(moment0_stencil.domain(), [&](auto stencil_elem) {
                    double const coeff = moment0_stencil.mem(stencil_elem);
                    if (coeff == 0.0) {
                        return;
                    }
                    auto const potential_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                    moment0_columns_host(row, moment0_count) = static_cast<int>(flat_index(
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimX>(potential_elem).uid()),
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimY>(potential_elem).uid())));
                    moment0_coefficients_host(row, moment0_count) = coeff;
                    ++moment0_count;
                });
                moment0_counts_host(row) = moment0_count;

                int moment1_count = 0;
                auto moment1_stencil
                        = MagneticVectorPotentialToMagneticInduction::template forward_value<Y>(
                                elem);
                ddc::device_for_each(moment1_stencil.domain(), [&](auto stencil_elem) {
                    double const coeff = moment1_stencil.mem(stencil_elem);
                    if (coeff == 0.0) {
                        return;
                    }
                    auto const potential_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                    moment1_columns_host(row, moment1_count) = static_cast<int>(flat_index(
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimX>(potential_elem).uid()),
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimY>(potential_elem).uid())));
                    moment1_coefficients_host(row, moment1_count) = coeff;
                    ++moment1_count;
                });
                moment1_counts_host(row) = moment1_count;
                transposed_moment0_counts_host(row) = 0;
                transposed_moment1_counts_host(row) = 0;

                auto outer_chain = sil::exterior::tangent_basis<1, NodeDomain2D>(elem);
                auto outer_lower_chain = sil::exterior::tangent_basis<0, NodeDomain2D>(elem);
                int outer0_count = 0;
                auto outer0_stencil = sil::exterior::TransposedCoboundary<
                        sil::tensor::Covariant<InPlaneIndex>,
                        ScalarPotentialIndex>::
                        value([](auto, auto) { return 0.0; },
                              outer_chain,
                              outer_lower_chain,
                              elem,
                              output_y);
                ddc::device_for_each(outer0_stencil.domain(), [&](auto stencil_elem) {
                    double const coeff = outer0_stencil.mem(stencil_elem);
                    if (coeff == 0.0) {
                        return;
                    }
                    auto const potential_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                    outer0_columns_host(row, outer0_count) = static_cast<int>(flat_index(
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimX>(potential_elem).uid()),
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimY>(potential_elem).uid())));
                    outer0_coefficients_host(row, outer0_count) = coeff;
                    ++outer0_count;
                });
                outer0_counts_host(row) = outer0_count;

                int outer1_count = 0;
                auto outer1_stencil = sil::exterior::TransposedCoboundary<
                        sil::tensor::Covariant<InPlaneIndex>,
                        ScalarPotentialIndex>::
                        value([](auto, auto) { return 0.0; },
                              outer_chain,
                              outer_lower_chain,
                              elem,
                              output_x);
                ddc::device_for_each(outer1_stencil.domain(), [&](auto stencil_elem) {
                    double const coeff = -outer1_stencil.mem(stencil_elem);
                    if (coeff == 0.0) {
                        return;
                    }
                    auto const potential_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                    outer1_columns_host(row, outer1_count) = static_cast<int>(flat_index(
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimX>(potential_elem).uid()),
                            static_cast<std::size_t>(
                                    ddc::DiscreteElement<DDimY>(potential_elem).uid())));
                    outer1_coefficients_host(row, outer1_count) = coeff;
                    ++outer1_count;
                });
                outer1_counts_host(row) = outer1_count;
            }
        }
        for (std::size_t sampled_row = 0; sampled_row < m_nx * m_ny; ++sampled_row) {
            std::size_t const sampled_i = sampled_row % m_nx;
            std::size_t const sampled_j = sampled_row / m_nx;
            if (sampled_i == 0 || sampled_j == 0 || sampled_i + 1 == m_nx
                || sampled_j + 1 == m_ny) {
                continue;
            }
            for (int slot = 0; slot < moment0_counts_host(sampled_row); ++slot) {
                std::size_t const row
                        = static_cast<std::size_t>(moment0_columns_host(sampled_row, slot));
                if (row >= m_nx * m_ny) {
                    continue;
                }
                int const count = transposed_moment0_counts_host(row);
                if (count >= TRANSPOSED_MOMENT_STENCIL_MAX_SIZE) {
                    throw std::runtime_error("transposed moment0 stencil capacity exceeded");
                }
                transposed_moment0_columns_host(row, count) = static_cast<int>(sampled_row);
                transposed_moment0_coefficients_host(row, count)
                        = moment0_coefficients_host(sampled_row, slot);
                transposed_moment0_counts_host(row) = count + 1;
            }
            for (int slot = 0; slot < moment1_counts_host(sampled_row); ++slot) {
                std::size_t const row
                        = static_cast<std::size_t>(moment1_columns_host(sampled_row, slot));
                if (row >= m_nx * m_ny) {
                    continue;
                }
                int const count = transposed_moment1_counts_host(row);
                if (count >= TRANSPOSED_MOMENT_STENCIL_MAX_SIZE) {
                    throw std::runtime_error("transposed moment1 stencil capacity exceeded");
                }
                transposed_moment1_columns_host(row, count) = static_cast<int>(sampled_row);
                transposed_moment1_coefficients_host(row, count)
                        = moment1_coefficients_host(sampled_row, slot);
                transposed_moment1_counts_host(row) = count + 1;
            }
        }
        Kokkos::deep_copy(m_moment0_columns, moment0_columns_host);
        Kokkos::deep_copy(m_moment0_coefficients, moment0_coefficients_host);
        Kokkos::deep_copy(m_moment1_columns, moment1_columns_host);
        Kokkos::deep_copy(m_moment1_coefficients, moment1_coefficients_host);
        Kokkos::deep_copy(m_moment0_counts, moment0_counts_host);
        Kokkos::deep_copy(m_moment1_counts, moment1_counts_host);
        Kokkos::deep_copy(m_transposed_moment0_columns, transposed_moment0_columns_host);
        Kokkos::deep_copy(m_transposed_moment0_coefficients, transposed_moment0_coefficients_host);
        Kokkos::deep_copy(m_transposed_moment1_columns, transposed_moment1_columns_host);
        Kokkos::deep_copy(m_transposed_moment1_coefficients, transposed_moment1_coefficients_host);
        Kokkos::deep_copy(m_transposed_moment0_counts, transposed_moment0_counts_host);
        Kokkos::deep_copy(m_transposed_moment1_counts, transposed_moment1_counts_host);
        Kokkos::deep_copy(m_outer0_columns, outer0_columns_host);
        Kokkos::deep_copy(m_outer0_coefficients, outer0_coefficients_host);
        Kokkos::deep_copy(m_outer1_columns, outer1_columns_host);
        Kokkos::deep_copy(m_outer1_coefficients, outer1_coefficients_host);
        Kokkos::deep_copy(m_outer0_counts, outer0_counts_host);
        Kokkos::deep_copy(m_outer1_counts, outer1_counts_host);
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t size() const
    {
        return m_nx * m_ny;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool is_boundary_node(std::size_t i, std::size_t j) const
    {
        return i == 0 || j == 0 || i + 1 == m_nx || j + 1 == m_ny;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        std::size_t const nx = m_nx;
        std::size_t const ny = m_ny;
        auto const moment0_columns = m_moment0_columns;
        auto const moment0_coefficients = m_moment0_coefficients;
        auto const moment1_columns = m_moment1_columns;
        auto const moment1_coefficients = m_moment1_coefficients;
        auto const moment0_counts = m_moment0_counts;
        auto const moment1_counts = m_moment1_counts;
        auto const transposed_moment0_columns = m_transposed_moment0_columns;
        auto const transposed_moment0_coefficients = m_transposed_moment0_coefficients;
        auto const transposed_moment1_columns = m_transposed_moment1_columns;
        auto const transposed_moment1_coefficients = m_transposed_moment1_coefficients;
        auto const transposed_moment0_counts = m_transposed_moment0_counts;
        auto const transposed_moment1_counts = m_transposed_moment1_counts;
        auto const outer0_columns = m_outer0_columns;
        auto const outer0_coefficients = m_outer0_coefficients;
        auto const outer1_columns = m_outer1_columns;
        auto const outer1_coefficients = m_outer1_coefficients;
        auto const outer0_counts = m_outer0_counts;
        auto const outer1_counts = m_outer1_counts;
        auto const equations = m_equations;
        auto const criterion = m_criterion;

        ddc::parallel_for_each(
                exec_space,
                m_node_domain,
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                    std::size_t const i
                            = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(elem).uid());
                    std::size_t const j
                            = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(elem).uid());
                    std::size_t const row = i + nx * j;
                    if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
                        output(row, 0) = input(row, 0);
                        return;
                    }

                    double residual = 0.0;
                    if (criterion == solvers::Criterion::PotentialTemporalDerivative
                        || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                        for (int slot = 0; slot < transposed_moment0_counts(row); ++slot) {
                            double const transpose_coefficient
                                    = transposed_moment0_coefficients(row, slot);
                            std::size_t const sampled_row = static_cast<std::size_t>(
                                    transposed_moment0_columns(row, slot));
                            auto const sampled_elem = ddc::DiscreteElement<
                                    DDimX,
                                    DDimY>(sampled_row % nx, sampled_row / nx);
                            double moment0 = 0.0;
                            for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                                moment0 += moment0_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment0_columns(sampled_row, k)),
                                                   0);
                            }
                            double moment1 = 0.0;
                            for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                                moment1 += moment1_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment1_columns(sampled_row, k)),
                                                   0);
                            }
                            MagneticMoments const moments {moment0, moment1, 0.0};
                            residual += transpose_coefficient * [&]() {
                                if constexpr (requires {
                                                  equations.template dpotential_dt<
                                                          X>(moments, sampled_elem);
                                              }) {
                                    return equations
                                            .template dpotential_dt<X>(moments, sampled_elem);
                                } else {
                                    return equations
                                            .template dpotential_dt<X>(moment0, sampled_elem);
                                }
                            }();
                        }
                        for (int slot = 0; slot < transposed_moment1_counts(row); ++slot) {
                            double const transpose_coefficient
                                    = transposed_moment1_coefficients(row, slot);
                            std::size_t const sampled_row = static_cast<std::size_t>(
                                    transposed_moment1_columns(row, slot));
                            auto const sampled_elem = ddc::DiscreteElement<
                                    DDimX,
                                    DDimY>(sampled_row % nx, sampled_row / nx);
                            double moment0 = 0.0;
                            for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                                moment0 += moment0_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment0_columns(sampled_row, k)),
                                                   0);
                            }
                            double moment1 = 0.0;
                            for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                                moment1 += moment1_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment1_columns(sampled_row, k)),
                                                   0);
                            }
                            MagneticMoments const moments {moment0, moment1, 0.0};
                            residual += transpose_coefficient * [&]() {
                                if constexpr (requires {
                                                  equations.template dpotential_dt<
                                                          Y>(moments, sampled_elem);
                                              }) {
                                    return equations
                                            .template dpotential_dt<Y>(moments, sampled_elem);
                                } else {
                                    return equations
                                            .template dpotential_dt<Y>(moment1, sampled_elem);
                                }
                            }();
                        }
                    }
                    if (criterion == solvers::Criterion::MomentsTemporalDerivative
                        || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                        for (int slot = 0; slot < outer0_counts(row); ++slot) {
                            double const outer_coefficient = outer0_coefficients(row, slot);
                            std::size_t const sampled_row
                                    = static_cast<std::size_t>(outer0_columns(row, slot));
                            auto const sampled_elem = ddc::DiscreteElement<
                                    DDimX,
                                    DDimY>(sampled_row % nx, sampled_row / nx);
                            double moment0 = 0.0;
                            for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                                moment0 += moment0_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment0_columns(sampled_row, k)),
                                                   0);
                            }
                            double moment1 = 0.0;
                            for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                                moment1 += moment1_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment1_columns(sampled_row, k)),
                                                   0);
                            }
                            MagneticMoments const moments {moment0, moment1, 0.0};
                            residual -= outer_coefficient * [&]() {
                                if constexpr (requires {
                                                  equations.template dpotential_dt<
                                                          X>(moments, sampled_elem);
                                              }) {
                                    return equations
                                            .template dpotential_dt<X>(moments, sampled_elem);
                                } else {
                                    return equations
                                            .template dpotential_dt<X>(moment0, sampled_elem);
                                }
                            }();
                        }
                        for (int slot = 0; slot < outer1_counts(row); ++slot) {
                            double const outer_coefficient = outer1_coefficients(row, slot);
                            std::size_t const sampled_row
                                    = static_cast<std::size_t>(outer1_columns(row, slot));
                            auto const sampled_elem = ddc::DiscreteElement<
                                    DDimX,
                                    DDimY>(sampled_row % nx, sampled_row / nx);
                            double moment0 = 0.0;
                            for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                                moment0 += moment0_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment0_columns(sampled_row, k)),
                                                   0);
                            }
                            double moment1 = 0.0;
                            for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                                moment1 += moment1_coefficients(sampled_row, k)
                                           * input(static_cast<std::size_t>(
                                                           moment1_columns(sampled_row, k)),
                                                   0);
                            }
                            MagneticMoments const moments {moment0, moment1, 0.0};
                            residual -= outer_coefficient * [&]() {
                                if constexpr (requires {
                                                  equations.template dpotential_dt<
                                                          Y>(moments, sampled_elem);
                                              }) {
                                    return equations
                                            .template dpotential_dt<Y>(moments, sampled_elem);
                                } else {
                                    return equations
                                            .template dpotential_dt<Y>(moment1, sampled_elem);
                                }
                            }();
                        }
                    }
                    output(row, 0) = residual;
                });
    }

    [[nodiscard]] Equations const& equations() const
    {
        return m_equations;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION coord_view_type x_coords() const
    {
        return m_x_coords;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION coord_view_type y_coords() const
    {
        return m_y_coords;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION solvers::Criterion criterion() const
    {
        return m_criterion;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto transposed_moment0_columns() const
    {
        return m_transposed_moment0_columns;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto transposed_moment0_coefficients() const
    {
        return m_transposed_moment0_coefficients;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto transposed_moment1_columns() const
    {
        return m_transposed_moment1_columns;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto transposed_moment1_coefficients() const
    {
        return m_transposed_moment1_coefficients;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto transposed_moment0_counts() const
    {
        return m_transposed_moment0_counts;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto transposed_moment1_counts() const
    {
        return m_transposed_moment1_counts;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto moment0_columns() const
    {
        return m_moment0_columns;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto moment0_coefficients() const
    {
        return m_moment0_coefficients;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto moment1_columns() const
    {
        return m_moment1_columns;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto moment1_coefficients() const
    {
        return m_moment1_coefficients;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto moment0_counts() const
    {
        return m_moment0_counts;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto moment1_counts() const
    {
        return m_moment1_counts;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto outer0_columns() const
    {
        return m_outer0_columns;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto outer0_coefficients() const
    {
        return m_outer0_coefficients;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto outer1_columns() const
    {
        return m_outer1_columns;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto outer1_coefficients() const
    {
        return m_outer1_coefficients;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto outer0_counts() const
    {
        return m_outer0_counts;
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION auto outer1_counts() const
    {
        return m_outer1_counts;
    }

private:
    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t flat_index(std::size_t i, std::size_t j) const
    {
        return i + m_nx * j;
    }
};

template <
        class ExecSpace,
        class MemorySpace,
        class Equations,
        class StateView,
        class InputView,
        class OutputView>
void apply_jacobian(
        ExecSpace exec_space,
        MagnetostaticsOperator2D<MemorySpace, Equations> const& operator_model,
        StateView state,
        InputView input,
        OutputView output)
{
    std::size_t const nx = operator_model.x_coords().extent(0);
    std::size_t const ny = operator_model.y_coords().extent(0);
    auto const moment0_columns = operator_model.moment0_columns();
    auto const moment0_coefficients = operator_model.moment0_coefficients();
    auto const moment1_columns = operator_model.moment1_columns();
    auto const moment1_coefficients = operator_model.moment1_coefficients();
    auto const moment0_counts = operator_model.moment0_counts();
    auto const moment1_counts = operator_model.moment1_counts();
    auto const transposed_moment0_columns = operator_model.transposed_moment0_columns();
    auto const transposed_moment0_coefficients = operator_model.transposed_moment0_coefficients();
    auto const transposed_moment1_columns = operator_model.transposed_moment1_columns();
    auto const transposed_moment1_coefficients = operator_model.transposed_moment1_coefficients();
    auto const transposed_moment0_counts = operator_model.transposed_moment0_counts();
    auto const transposed_moment1_counts = operator_model.transposed_moment1_counts();
    auto const outer0_columns = operator_model.outer0_columns();
    auto const outer0_coefficients = operator_model.outer0_coefficients();
    auto const outer1_columns = operator_model.outer1_columns();
    auto const outer1_coefficients = operator_model.outer1_coefficients();
    auto const outer0_counts = operator_model.outer0_counts();
    auto const outer1_counts = operator_model.outer1_counts();
    auto const equations = operator_model.equations();
    auto const criterion = operator_model.criterion();
    auto const node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(nx, ny));

    ddc::parallel_for_each(
            exec_space,
            node_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                std::size_t const i
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(elem).uid());
                std::size_t const j
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(elem).uid());
                std::size_t const row = i + nx * j;
                if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
                    output(row, 0) = input(row, 0);
                    return;
                }

                double residual = 0.0;
                auto add_sampled_contribution = [&](double row_coefficient,
                                                    std::size_t sampled_row,
                                                    bool use_first_component) {
                    auto const sampled_elem = ddc::
                            DiscreteElement<DDimX, DDimY>(sampled_row % nx, sampled_row / nx);
                    double state_moment0 = 0.0;
                    double delta_moment0 = 0.0;
                    for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                        std::size_t const column
                                = static_cast<std::size_t>(moment0_columns(sampled_row, k));
                        state_moment0 += moment0_coefficients(sampled_row, k) * state(column, 0);
                        delta_moment0 += moment0_coefficients(sampled_row, k) * input(column, 0);
                    }
                    double state_moment1 = 0.0;
                    double delta_moment1 = 0.0;
                    for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                        std::size_t const column
                                = static_cast<std::size_t>(moment1_columns(sampled_row, k));
                        state_moment1 += moment1_coefficients(sampled_row, k) * state(column, 0);
                        delta_moment1 += moment1_coefficients(sampled_row, k) * input(column, 0);
                    }
                    MagneticMoments const moments {state_moment0, state_moment1, 0.0};
                    double const h00 = equations.template jacobian<X, X>(moments, sampled_elem);
                    double const h01 = equations.template jacobian<X, Y>(moments, sampled_elem);
                    double const h10 = equations.template jacobian<Y, X>(moments, sampled_elem);
                    double const h11 = equations.template jacobian<Y, Y>(moments, sampled_elem);
                    residual += row_coefficient
                                * (use_first_component
                                           ? (h00 * delta_moment0 + h01 * delta_moment1)
                                           : (h10 * delta_moment0 + h11 * delta_moment1));
                };

                if (criterion == solvers::Criterion::PotentialTemporalDerivative
                    || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                    for (int slot = 0; slot < transposed_moment0_counts(row); ++slot) {
                        add_sampled_contribution(
                                transposed_moment0_coefficients(row, slot),
                                static_cast<std::size_t>(transposed_moment0_columns(row, slot)),
                                true);
                    }
                    for (int slot = 0; slot < transposed_moment1_counts(row); ++slot) {
                        add_sampled_contribution(
                                transposed_moment1_coefficients(row, slot),
                                static_cast<std::size_t>(transposed_moment1_columns(row, slot)),
                                false);
                    }
                }
                if (criterion == solvers::Criterion::MomentsTemporalDerivative
                    || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                    for (int slot = 0; slot < outer0_counts(row); ++slot) {
                        add_sampled_contribution(
                                -outer0_coefficients(row, slot),
                                static_cast<std::size_t>(outer0_columns(row, slot)),
                                true);
                    }
                    for (int slot = 0; slot < outer1_counts(row); ++slot) {
                        add_sampled_contribution(
                                -outer1_coefficients(row, slot),
                                static_cast<std::size_t>(outer1_columns(row, slot)),
                                false);
                    }
                }
                output(row, 0) = residual;
            });
}

template <class MemorySpace, class Equations, class StateView>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        MagnetostaticsOperator2D<MemorySpace, Equations> const& operator_model,
        StateView state)
{
    std::size_t const size = operator_model.size();
    Kokkos::DefaultExecutionSpace exec_space;
    Kokkos::View<double* [64]> coefficients("similie_magnetostatics_matrix_coefficients", size);
    Kokkos::View<int* [64]> columns("similie_magnetostatics_matrix_columns", size);
    Kokkos::View<int*> counts("similie_magnetostatics_matrix_counts", size);

    auto equations = operator_model.equations();
    auto const criterion = operator_model.criterion();
    std::size_t const nx = operator_model.x_coords().extent(0);
    std::size_t const ny = operator_model.y_coords().extent(0);
    auto const transposed_moment0_columns = operator_model.transposed_moment0_columns();
    auto const transposed_moment0_coefficients = operator_model.transposed_moment0_coefficients();
    auto const transposed_moment1_columns = operator_model.transposed_moment1_columns();
    auto const transposed_moment1_coefficients = operator_model.transposed_moment1_coefficients();
    auto const transposed_moment0_counts = operator_model.transposed_moment0_counts();
    auto const transposed_moment1_counts = operator_model.transposed_moment1_counts();
    auto const moment0_columns = operator_model.moment0_columns();
    auto const moment0_coefficients = operator_model.moment0_coefficients();
    auto const moment1_columns = operator_model.moment1_columns();
    auto const moment1_coefficients = operator_model.moment1_coefficients();
    auto const moment0_counts = operator_model.moment0_counts();
    auto const moment1_counts = operator_model.moment1_counts();
    auto const outer0_columns = operator_model.outer0_columns();
    auto const outer0_coefficients = operator_model.outer0_coefficients();
    auto const outer1_columns = operator_model.outer1_columns();
    auto const outer1_coefficients = operator_model.outer1_coefficients();
    auto const outer0_counts = operator_model.outer0_counts();
    auto const outer1_counts = operator_model.outer1_counts();
    auto node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(nx, ny));

    ddc::parallel_for_each(
            "similie_assemble_magnetostatics_jacobian",
            exec_space,
            node_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                std::size_t const i
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(elem).uid());
                std::size_t const j
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(elem).uid());
                std::size_t const row = i + nx * j;

                auto set_entry = [&](int slot, std::size_t column, double value) {
                    columns(row, slot) = static_cast<int>(column);
                    coefficients(row, slot) = value;
                };
                auto add_entry = [&](int& count, std::size_t column, double value) {
                    for (int slot = 0; slot < count; ++slot) {
                        if (columns(row, slot) == static_cast<int>(column)) {
                            coefficients(row, slot) += value;
                            return;
                        }
                    }
                    if (count >= 64) {
                        Kokkos::abort("magnetostatics matrix row capacity exceeded");
                    }
                    columns(row, count) = static_cast<int>(column);
                    coefficients(row, count) = value;
                    ++count;
                };

                for (int slot = 0; slot < 64; ++slot) {
                    set_entry(slot, row, 0.0);
                }

                if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
                    counts(row) = 1;
                    set_entry(0, row, 1.0);
                    return;
                }

                int count = 0;
                auto add_sampled_block = [&](double row_coefficient,
                                             std::size_t sampled_row,
                                             bool use_first_component) {
                    auto const sampled_elem = ddc::
                            DiscreteElement<DDimX, DDimY>(sampled_row % nx, sampled_row / nx);
                    double state_moment0 = 0.0;
                    for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                        state_moment0 += moment0_coefficients(sampled_row, k)
                                         * state(static_cast<std::size_t>(
                                                         moment0_columns(sampled_row, k)),
                                                 0);
                    }
                    double state_moment1 = 0.0;
                    for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                        state_moment1 += moment1_coefficients(sampled_row, k)
                                         * state(static_cast<std::size_t>(
                                                         moment1_columns(sampled_row, k)),
                                                 0);
                    }
                    MagneticMoments const moments {state_moment0, state_moment1, 0.0};
                    double const h00 = equations.template jacobian<X, X>(moments, sampled_elem);
                    double const h01 = equations.template jacobian<X, Y>(moments, sampled_elem);
                    double const h10 = equations.template jacobian<Y, X>(moments, sampled_elem);
                    double const h11 = equations.template jacobian<Y, Y>(moments, sampled_elem);
                    for (int k = 0; k < moment0_counts(sampled_row); ++k) {
                        std::size_t const column
                                = static_cast<std::size_t>(moment0_columns(sampled_row, k));
                        double const value
                                = row_coefficient
                                  * (use_first_component
                                             ? h00 * moment0_coefficients(sampled_row, k)
                                             : h10 * moment0_coefficients(sampled_row, k));
                        if (value != 0.0) {
                            add_entry(count, column, value);
                        }
                    }
                    for (int k = 0; k < moment1_counts(sampled_row); ++k) {
                        std::size_t const column
                                = static_cast<std::size_t>(moment1_columns(sampled_row, k));
                        double const value
                                = row_coefficient
                                  * (use_first_component
                                             ? h01 * moment1_coefficients(sampled_row, k)
                                             : h11 * moment1_coefficients(sampled_row, k));
                        if (value != 0.0) {
                            add_entry(count, column, value);
                        }
                    }
                };

                if (criterion == solvers::Criterion::PotentialTemporalDerivative
                    || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                    for (int slot = 0; slot < transposed_moment0_counts(row); ++slot) {
                        add_sampled_block(
                                transposed_moment0_coefficients(row, slot),
                                static_cast<std::size_t>(transposed_moment0_columns(row, slot)),
                                true);
                    }
                    for (int slot = 0; slot < transposed_moment1_counts(row); ++slot) {
                        add_sampled_block(
                                transposed_moment1_coefficients(row, slot),
                                static_cast<std::size_t>(transposed_moment1_columns(row, slot)),
                                false);
                    }
                }
                if (criterion == solvers::Criterion::MomentsTemporalDerivative
                    || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                    for (int slot = 0; slot < outer0_counts(row); ++slot) {
                        add_sampled_block(
                                -outer0_coefficients(row, slot),
                                static_cast<std::size_t>(outer0_columns(row, slot)),
                                true);
                    }
                    for (int slot = 0; slot < outer1_counts(row); ++slot) {
                        add_sampled_block(
                                -outer1_coefficients(row, slot),
                                static_cast<std::size_t>(outer1_columns(row, slot)),
                                false);
                    }
                }
                counts(row) = count;
            });
    exec_space.fence();

    auto coefficients_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficients);
    auto columns_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), columns);
    auto counts_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);

    gko::matrix_data<double, gko::int32> matrix_data(gko::dim<2>(size, size));
    matrix_data.nonzeros.reserve(size * 64);
    for (std::size_t row = 0; row < size; ++row) {
        for (int slot = 0; slot < counts_host(row); ++slot) {
            double const coefficient = coefficients_host(row, slot);
            if (coefficient == 0.0) {
                continue;
            }
            matrix_data.nonzeros.emplace_back(
                    row,
                    static_cast<std::size_t>(columns_host(row, slot)),
                    coefficient);
        }
    }
    return matrix_data;
}

template <class MemorySpace, class Equations>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        MagnetostaticsOperator2D<MemorySpace, Equations> const& operator_model)
{
    std::size_t const size = operator_model.size();
    Kokkos::DefaultExecutionSpace exec_space;
    Kokkos::View<double* [64]> coefficients("similie_magnetostatics_matrix_coefficients", size);
    Kokkos::View<int* [64]> columns("similie_magnetostatics_matrix_columns", size);
    Kokkos::View<int*> counts("similie_magnetostatics_matrix_counts", size);

    auto equations = operator_model.equations();
    auto const criterion = operator_model.criterion();
    std::size_t const nx = operator_model.x_coords().extent(0);
    std::size_t const ny = operator_model.y_coords().extent(0);
    auto const transposed_moment0_columns = operator_model.transposed_moment0_columns();
    auto const transposed_moment0_coefficients = operator_model.transposed_moment0_coefficients();
    auto const transposed_moment1_columns = operator_model.transposed_moment1_columns();
    auto const transposed_moment1_coefficients = operator_model.transposed_moment1_coefficients();
    auto const transposed_moment0_counts = operator_model.transposed_moment0_counts();
    auto const transposed_moment1_counts = operator_model.transposed_moment1_counts();
    auto const moment0_columns = operator_model.moment0_columns();
    auto const moment0_coefficients = operator_model.moment0_coefficients();
    auto const moment1_columns = operator_model.moment1_columns();
    auto const moment1_coefficients = operator_model.moment1_coefficients();
    auto const moment0_counts = operator_model.moment0_counts();
    auto const moment1_counts = operator_model.moment1_counts();
    auto node_domain = ddc::DiscreteDomain<DDimX, DDimY>(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(nx, ny));
    using OutputIndex = sil::exterior::
            coboundary_index_t<sil::tensor::Covariant<InPlaneIndex>, ScalarPotentialIndex>;
    auto const output_y = sil::tensor::TensorAccessor<OutputIndex>().template access_element<Y>();
    auto const output_x = sil::tensor::TensorAccessor<OutputIndex>().template access_element<X>();

    ddc::parallel_for_each(
            "similie_assemble_magnetostatics_matrix",
            exec_space,
            node_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> elem) {
                std::size_t const i
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimX>(elem).uid());
                std::size_t const j
                        = static_cast<std::size_t>(ddc::DiscreteElement<DDimY>(elem).uid());
                std::size_t const row = i + nx * j;

                auto set_entry = [&](int slot, std::size_t column, double value) {
                    columns(row, slot) = static_cast<int>(column);
                    coefficients(row, slot) = value;
                };
                auto add_entry = [&](int& count, std::size_t column, double value) {
                    for (int slot = 0; slot < count; ++slot) {
                        if (columns(row, slot) == static_cast<int>(column)) {
                            coefficients(row, slot) += value;
                            return;
                        }
                    }
                    if (count >= 64) {
                        Kokkos::abort("magnetostatics matrix row capacity exceeded");
                    }
                    columns(row, count) = static_cast<int>(column);
                    coefficients(row, count) = value;
                    ++count;
                };

                for (int slot = 0; slot < 64; ++slot) {
                    set_entry(slot, row, 0.0);
                }

                if (i == 0 || j == 0 || i + 1 == nx || j + 1 == ny) {
                    counts(row) = 1;
                    set_entry(0, row, 1.0);
                    return;
                }

                int count = 0;
                if (criterion == solvers::Criterion::PotentialTemporalDerivative
                    || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                    for (int slot = 0; slot < transposed_moment0_counts(row); ++slot) {
                        double const transpose_coefficient
                                = transposed_moment0_coefficients(row, slot);
                        std::size_t const sampled_row
                                = static_cast<std::size_t>(transposed_moment0_columns(row, slot));
                        auto const sampled_elem = ddc::
                                DiscreteElement<DDimX, DDimY>(sampled_row % nx, sampled_row / nx);
                        auto moments_stencil
                                = equations.template dpotential_dt_value<X>(sampled_elem);
                        ddc::device_for_each(moments_stencil.domain(), [&](auto moments_elem) {
                            double const value
                                    = transpose_coefficient * moments_stencil.mem(moments_elem);
                            if (value == 0.0) {
                                return;
                            }
                            auto const potential_elem
                                    = ddc::DiscreteElement<DDimX, DDimY>(moments_elem);
                            std::size_t const column
                                    = static_cast<std::size_t>(
                                              ddc::DiscreteElement<DDimX>(potential_elem).uid())
                                      + nx
                                                * static_cast<std::size_t>(
                                                        ddc::DiscreteElement<DDimY>(potential_elem)
                                                                .uid());
                            add_entry(count, column, value);
                        });
                    }
                    for (int slot = 0; slot < transposed_moment1_counts(row); ++slot) {
                        double const transpose_coefficient
                                = transposed_moment1_coefficients(row, slot);
                        std::size_t const sampled_row
                                = static_cast<std::size_t>(transposed_moment1_columns(row, slot));
                        auto const sampled_elem = ddc::
                                DiscreteElement<DDimX, DDimY>(sampled_row % nx, sampled_row / nx);
                        auto moments_stencil
                                = equations.template dpotential_dt_value<Y>(sampled_elem);
                        ddc::device_for_each(moments_stencil.domain(), [&](auto moments_elem) {
                            double const value
                                    = transpose_coefficient * moments_stencil.mem(moments_elem);
                            if (value == 0.0) {
                                return;
                            }
                            auto const potential_elem
                                    = ddc::DiscreteElement<DDimX, DDimY>(moments_elem);
                            std::size_t const column
                                    = static_cast<std::size_t>(
                                              ddc::DiscreteElement<DDimX>(potential_elem).uid())
                                      + nx
                                                * static_cast<std::size_t>(
                                                        ddc::DiscreteElement<DDimY>(potential_elem)
                                                                .uid());
                            add_entry(count, column, value);
                        });
                    }
                }
                if (criterion == solvers::Criterion::MomentsTemporalDerivative
                    || criterion == solvers::Criterion::PotentialAndMomentsTemporalDerivative) {
                    auto outer_chain = sil::exterior::tangent_basis<1, NodeDomain2D>(elem);
                    auto outer_lower_chain = sil::exterior::tangent_basis<0, NodeDomain2D>(elem);
                    auto outer0_stencil = sil::exterior::TransposedCoboundary<
                            sil::tensor::Covariant<InPlaneIndex>,
                            ScalarPotentialIndex>::
                            value([](auto, auto) { return 0.0; },
                                  outer_chain,
                                  outer_lower_chain,
                                  elem,
                                  output_y);
                    ddc::device_for_each(outer0_stencil.domain(), [&](auto stencil_elem) {
                        double const outer_coeff = outer0_stencil.mem(stencil_elem);
                        if (outer_coeff == 0.0) {
                            return;
                        }
                        auto const sampled_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                        auto moments_stencil
                                = equations.template dpotential_dt_value<X>(sampled_elem);
                        ddc::device_for_each(moments_stencil.domain(), [&](auto moments_elem) {
                            double const value = -outer_coeff * moments_stencil.mem(moments_elem);
                            if (value == 0.0) {
                                return;
                            }
                            auto const potential_elem
                                    = ddc::DiscreteElement<DDimX, DDimY>(moments_elem);
                            std::size_t const column
                                    = static_cast<std::size_t>(
                                              ddc::DiscreteElement<DDimX>(potential_elem).uid())
                                      + nx
                                                * static_cast<std::size_t>(
                                                        ddc::DiscreteElement<DDimY>(potential_elem)
                                                                .uid());
                            add_entry(count, column, value);
                        });
                    });
                    auto outer1_stencil = sil::exterior::TransposedCoboundary<
                            sil::tensor::Covariant<InPlaneIndex>,
                            ScalarPotentialIndex>::
                            value([](auto, auto) { return 0.0; },
                                  outer_chain,
                                  outer_lower_chain,
                                  elem,
                                  output_x);
                    ddc::device_for_each(outer1_stencil.domain(), [&](auto stencil_elem) {
                        double const outer_coeff = -outer1_stencil.mem(stencil_elem);
                        if (outer_coeff == 0.0) {
                            return;
                        }
                        auto const sampled_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
                        auto moments_stencil
                                = equations.template dpotential_dt_value<Y>(sampled_elem);
                        ddc::device_for_each(moments_stencil.domain(), [&](auto moments_elem) {
                            double const value = -outer_coeff * moments_stencil.mem(moments_elem);
                            if (value == 0.0) {
                                return;
                            }
                            auto const potential_elem
                                    = ddc::DiscreteElement<DDimX, DDimY>(moments_elem);
                            std::size_t const column
                                    = static_cast<std::size_t>(
                                              ddc::DiscreteElement<DDimX>(potential_elem).uid())
                                      + nx
                                                * static_cast<std::size_t>(
                                                        ddc::DiscreteElement<DDimY>(potential_elem)
                                                                .uid());
                            add_entry(count, column, value);
                        });
                    });
                }
                counts(row) = count;
            });
    exec_space.fence();

    auto coefficients_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficients);
    auto columns_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), columns);
    auto counts_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);

    gko::matrix_data<double, gko::int32> matrix_data(gko::dim<2>(size, size));
    matrix_data.nonzeros.reserve(size * 64);
    for (std::size_t row = 0; row < size; ++row) {
        for (int slot = 0; slot < counts_host(row); ++slot) {
            double const coefficient = coefficients_host(row, slot);
            if (coefficient != 0.0) {
                matrix_data.nonzeros.emplace_back(
                        static_cast<gko::int32>(row),
                        static_cast<gko::int32>(columns_host(row, slot)),
                        coefficient);
            }
        }
    }
    return matrix_data;
}

} // namespace magnetostatics_local

using magnetostatics_local::X;
using magnetostatics_local::Y;
using magnetostatics_local::Z;

template <class... CDim>
using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<CDim...>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<CDim...>>>;

using PositionIndex2D = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;

using MetricIndex2D = MetricIndex<X, Y>;

using InPlaneInductionFormIndex = sil::tensor::Covariant<magnetostatics_local::InPlaneIndex>;
using InPlaneInductionIndexSeq = sil::tensor::upper_t<
        ddc::to_type_seq_t<sil::tensor::natural_domain_t<InPlaneInductionFormIndex>>>;

template <class Index, class NodeValueGetter>
double magnetic_induction_moment_from_potential_z(
        ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY> elem,
        NodeValueGetter&& node_value_z)
{
    using namespace magnetostatics_local;

    auto apply_stencil = [&](auto stencil) {
        double value = 0.0;
        ddc::host_for_each(stencil.domain(), [&](auto stencil_elem) {
            auto const potential_elem = ddc::DiscreteElement<DDimX, DDimY>(stencil_elem);
            value += stencil.mem(stencil_elem)
                     * node_value_z(
                             static_cast<std::size_t>(
                                     ddc::DiscreteElement<DDimX>(potential_elem).uid()),
                             static_cast<std::size_t>(
                                     ddc::DiscreteElement<DDimY>(potential_elem).uid()));
        });
        return value;
    };

    return apply_stencil(
            magnetostatics_local::MagneticVectorPotentialToMagneticInduction::
                    template forward_value<Index>(elem));
}

template <
        class ReadNodePosition,
        class ReadMu,
        class ReadNonlinearMaterial,
        class NonlinearConstitutiveLaw,
        class NodeValueGetter,
        class WriteCellOutput>
void fill_post_process_fields_on_cell_domain(
        ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY> const&
                cell_domain,
        ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY> const&
                node_domain,
        ReadNodePosition&& read_node_position,
        ReadMu&& read_mu,
        ReadNonlinearMaterial&& read_nonlinear_material,
        NonlinearConstitutiveLaw const& nonlinear_constitutive_law,
        NodeValueGetter&& node_value_z,
        WriteCellOutput&& write_cell_output)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex2D> position_accessor;
    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY, PositionIndex2D>
            position_dom(node_domain, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex2D> metric_accessor;
    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY, MetricIndex2D>
            metric_dom(node_domain, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    ddc::host_for_each(node_domain, [&](auto node_elem) {
        std::array<double, 2> const coordinates = read_node_position(node_elem);
        position(node_elem, position.accessor().template access_element<X>()) = coordinates[0];
        position(node_elem, position.accessor().template access_element<Y>()) = coordinates[1];

        metric(node_elem, metric.accessor().template access_element<X, X>()) = 1.0;
        metric(node_elem, metric.accessor().template access_element<X, Y>()) = 0.0;
        metric(node_elem, metric.accessor().template access_element<Y, Y>()) = 1.0;
    });

    ddc::host_for_each(cell_domain, [&](auto elem) {
        std::array<double, InPlaneInductionFormIndex::access_size()> reduced_induction_alloc {};
        [[maybe_unused]] sil::tensor::TensorAccessor<InPlaneInductionFormIndex>
                reduced_induction_accessor;
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<InPlaneInductionFormIndex>,
                Kokkos::layout_right,
                Kokkos::HostSpace>
                reduced_induction_span(
                        reduced_induction_alloc.data(),
                        reduced_induction_accessor.domain());
        sil::tensor::Tensor reduced_induction(reduced_induction_span);
        reduced_induction(reduced_induction.accessor().template access_element<X>())
                = magnetic_induction_moment_from_potential_z<Y>(elem, node_value_z);
        reduced_induction(reduced_induction.accessor().template access_element<Y>())
                = magnetic_induction_moment_from_potential_z<X>(elem, node_value_z);

        std::array<double, InPlaneInductionFormIndex::access_size()>
                reconstructed_induction_alloc {};
        [[maybe_unused]] sil::tensor::TensorAccessor<InPlaneInductionFormIndex>
                reconstructed_induction_accessor;
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<InPlaneInductionFormIndex>,
                Kokkos::layout_right,
                Kokkos::HostSpace>
                reconstructed_induction_span(
                        reconstructed_induction_alloc.data(),
                        reconstructed_induction_accessor.domain());
        sil::tensor::Tensor reconstructed_induction(reconstructed_induction_span);
        sil::exterior::Reconstruction<
                InPlaneInductionIndexSeq,
                decltype(position),
                ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>>::
                run(reconstructed_induction, reduced_induction, position, elem);

        std::array<double, 3> const magnetic_induction {
                reconstructed_induction(
                        reconstructed_induction.accessor().template access_element<Y>()),
                reconstructed_induction(
                        reconstructed_induction.accessor().template access_element<X>()),
                0.0,
        };
        std::array<double, 3> magnetic_field {0.0, 0.0, 0.0};
        std::array<double, 3> const unit_hodge {1.0, 1.0, 1.0};
        if (read_nonlinear_material(elem)) {
            magnetic_field = nonlinear_constitutive_law(
                    std::span<double const, 3>(unit_hodge.data(), unit_hodge.size()),
                    std::span<
                            double const,
                            3>(magnetic_induction.data(), magnetic_induction.size()));
        } else {
            physics::magnetostatics::LinearMagneticInductionToMagneticField const constitutive_law(
                    read_mu(elem));
            magnetic_field = {
                    constitutive_law(unit_hodge[0], magnetic_induction[0]),
                    constitutive_law(unit_hodge[1], magnetic_induction[1]),
                    0.0,
            };
        }

        write_cell_output(elem, magnetic_induction, magnetic_field);
    });
}

inline CellPostProcessFields make_cell_post_process_fields(
        std::array<double, 3> const& magnetic_induction,
        std::array<double, 3> const& magnetic_field)
{
    CellPostProcessFields cell_output {};
    cell_output.magnetic_induction = magnetic_induction;
    cell_output.magnetic_field = magnetic_field;
    double const half_trace = 0.5
                              * (magnetic_induction[0] * magnetic_field[0]
                                 + magnetic_induction[1] * magnetic_field[1]
                                 + magnetic_induction[2] * magnetic_field[2]);
    cell_output.maxwell_stress = {
            magnetic_induction[0] * magnetic_field[0] - half_trace,
            magnetic_induction[1] * magnetic_field[1] - half_trace,
            magnetic_induction[2] * magnetic_field[2] - half_trace,
            magnetic_induction[0] * magnetic_field[1],
            magnetic_induction[0] * magnetic_field[2],
            magnetic_induction[1] * magnetic_field[2],
    };
    return cell_output;
}

template <class FillStress, class ReadPosition, class WriteForceDensity>
void fill_force_density_on_cell_domain(
        ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY> const&
                cell_domain,
        FillStress&& fill_stress,
        ReadPosition&& read_position,
        WriteForceDensity&& write_force_density)
{
    using InPlaneOneFormIndex = sil::tensor::Covariant<magnetostatics_local::InPlaneIndex>;
    using ForceDensityIndex = physics::magnetostatics::ForceDensityIndex<X, Y, Z>;
    using ScalarIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;

    [[maybe_unused]] sil::tensor::TensorAccessor<ForceDensityIndex> force_density_accessor;
    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY, ForceDensityIndex>
            force_density_dom(cell_domain, force_density_accessor.domain());
    ddc::Chunk force_density_alloc(force_density_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor force_density_tensor(force_density_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex2D> metric_accessor;
    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY, MetricIndex2D>
            metric_dom(cell_domain, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex2D> position_accessor;
    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY, PositionIndex2D>
            position_dom(cell_domain, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<InPlaneOneFormIndex> one_form_accessor;
    ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY,
            InPlaneOneFormIndex>
            one_form_dom(cell_domain, one_form_accessor.domain());
    ddc::Chunk one_form_alloc(one_form_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor one_form_tensor(one_form_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<ScalarIndex> scalar_accessor;
    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY, ScalarIndex>
            scalar_dom(cell_domain, scalar_accessor.domain());
    ddc::Chunk scalar_alloc(scalar_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor scalar_tensor(scalar_alloc);

    ddc::host_for_each(cell_domain, [&](auto elem) {
        std::array<double, 2> const coordinates = read_position(elem);
        position(elem, position.accessor().template access_element<X>()) = coordinates[0];
        position(elem, position.accessor().template access_element<Y>()) = coordinates[1];

        metric(elem, metric.accessor().template access_element<X, X>()) = 1.0;
        metric(elem, metric.accessor().template access_element<X, Y>()) = 0.0;
        metric(elem, metric.accessor().template access_element<Y, Y>()) = 1.0;
    });

    auto staged_codifferential = sil::exterior::
            make_staged_codifferential<MetricIndex2D, InPlaneOneFormIndex, InPlaneOneFormIndex>(
                    Kokkos::DefaultHostExecutionSpace(),
                    one_form_tensor,
                    metric,
                    position);

    auto fill_force_component = [&](auto select_components, auto assign_output) {
        ddc::host_for_each(cell_domain, [&](auto elem) {
            std::array<double, 6> const stress = fill_stress(elem);
            std::array<double, 2> const one_form = select_components(stress);
            one_form_tensor(elem, one_form_tensor.accessor().template access_element<X>())
                    = one_form[0];
            one_form_tensor(elem, one_form_tensor.accessor().template access_element<Y>())
                    = one_form[1];
        });
        staged_codifferential.run(scalar_tensor, one_form_tensor);
        ddc::host_for_each(cell_domain, [&](auto elem) {
            assign_output(elem, -scalar_tensor(elem, ddc::DiscreteElement<ScalarIndex>(0)));
        });
    };

    fill_force_component(
            [](std::array<double, 6> const& stress) {
                return std::array<double, 2> {stress[0], stress[3]};
            },
            [&](auto elem, double value) {
                force_density_tensor(
                        elem,
                        force_density_tensor.accessor().template access_element<X>())
                        = value;
            });
    fill_force_component(
            [](std::array<double, 6> const& stress) {
                return std::array<double, 2> {stress[3], stress[1]};
            },
            [&](auto elem, double value) {
                force_density_tensor(
                        elem,
                        force_density_tensor.accessor().template access_element<Y>())
                        = value;
            });
    fill_force_component(
            [](std::array<double, 6> const& stress) {
                return std::array<double, 2> {stress[4], stress[5]};
            },
            [&](auto elem, double value) {
                force_density_tensor(
                        elem,
                        force_density_tensor.accessor().template access_element<Z>())
                        = value;
            });

    ddc::host_for_each(cell_domain, [&](auto elem) {
        write_force_density(
                elem,
                std::array<double, 3> {
                        force_density_tensor(
                                elem,
                                force_density_tensor.accessor().template access_element<X>()),
                        force_density_tensor(
                                elem,
                                force_density_tensor.accessor().template access_element<Y>()),
                        force_density_tensor(
                                elem,
                                force_density_tensor.accessor().template access_element<Z>()),
                });
    });
}

inline void fill_force_density_on_quadrilateral_grid(
        sil::onelab_interface::gmsh::StructuredGrid2D const& grid,
        std::vector<CellPostProcessFields>& cell_outputs)
{
    auto const cell_domain = ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY>(
            ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(0, 0),
            ddc::DiscreteVector<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY>(grid.ncell_x(), grid.ncell_y()));
    fill_force_density_on_cell_domain(
            cell_domain,
            [&](auto elem) {
                std::size_t const i = static_cast<std::size_t>(
                        ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                std::size_t const j = static_cast<std::size_t>(
                        ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                return cell_outputs[grid.cell_index(i, j)].maxwell_stress;
            },
            [&](auto elem) {
                std::size_t const i = static_cast<std::size_t>(
                        ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                std::size_t const j = static_cast<std::size_t>(
                        ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                return std::array<double, 2> {
                        grid.cell_center_x(i),
                        grid.cell_center_y(j),
                };
            },
            [&](auto elem, std::array<double, 3> const& force_density) {
                std::size_t const i = static_cast<std::size_t>(
                        ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                std::size_t const j = static_cast<std::size_t>(
                        ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                cell_outputs[grid.cell_index(i, j)].force_density = force_density;
            });
}

inline void fill_force_density_on_hexahedral_grid_xy_slices(
        sil::onelab_interface::gmsh::StructuredGrid3D const& grid,
        std::vector<CellPostProcessFields>& cell_outputs)
{
    auto const cell_domain = ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY>(
            ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(0, 0),
            ddc::DiscreteVector<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY>(grid.ncell_x(), grid.ncell_y()));
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        fill_force_density_on_cell_domain(
                cell_domain,
                [&](auto elem) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                    return cell_outputs[grid.cell_index(i, j, k)].maxwell_stress;
                },
                [&](auto elem) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                    return std::array<double, 2> {
                            grid.cell_center_x(i),
                            grid.cell_center_y(j),
                    };
                },
                [&](auto elem, std::array<double, 3> const& force_density) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                    cell_outputs[grid.cell_index(i, j, k)].force_density = force_density;
                });
    }
}

inline void write_results_view(
        std::filesystem::path const& output_view_file,
        sil::onelab_interface::gmsh::StructuredGrid2D const& grid,
        std::vector<CellInputFields> const& cell_inputs,
        std::vector<CellPostProcessFields> const& cell_outputs,
        std::vector<double> const& magnetic_vector_potential)
{
    std::ofstream stream(output_view_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_view_file.string());
    }

    stream << "View \"SimiLie linear magnetostatics permeability\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            auto const& cell_input = cell_inputs[grid.cell_index(i, j)];
            stream << "SP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                   << grid.z_value << "){" << cell_input.mu << "};\n";
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics current density\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            auto const& cell_input = cell_inputs[grid.cell_index(i, j)];
            stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                   << grid.z_value << "){" << cell_input.current_density[0] << ","
                   << cell_input.current_density[1] << "," << cell_input.current_density[2]
                   << "};\n";
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic induction\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            auto const& cell_output = cell_outputs[grid.cell_index(i, j)];
            stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                   << grid.z_value << "){" << cell_output.magnetic_induction[0] << ","
                   << cell_output.magnetic_induction[1] << "," << cell_output.magnetic_induction[2]
                   << "};\n";
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic field\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            auto const& cell_output = cell_outputs[grid.cell_index(i, j)];
            stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                   << grid.z_value << "){" << cell_output.magnetic_field[0] << ","
                   << cell_output.magnetic_field[1] << "," << cell_output.magnetic_field[2]
                   << "};\n";
        }
    }
    stream << "};\n";

    auto write_stress_view = [&](std::string_view view_name, std::size_t component) {
        stream << "View \"" << view_name << "\" {\n";
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto const& cell_output = cell_outputs[grid.cell_index(i, j)];
                stream << "SP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                       << grid.z_value << "){" << cell_output.maxwell_stress[component] << "};\n";
            }
        }
        stream << "};\n";
    };
    write_stress_view("SimiLie linear magnetostatics Maxwell stress xx", 0);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress yy", 1);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress zz", 2);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress xy", 3);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress xz", 4);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress yz", 5);

    stream << "View \"SimiLie linear magnetostatics force density\" {\n";
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            auto const& cell_output = cell_outputs[grid.cell_index(i, j)];
            stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                   << grid.z_value << "){" << cell_output.force_density[0] << ","
                   << cell_output.force_density[1] << "," << cell_output.force_density[2] << "};\n";
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic vector potential\" {\n";
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index = grid.node_index(i, j);
            stream << "VP(" << grid.x_coords[i] << "," << grid.y_coords[j] << "," << grid.z_value
                   << "){0,0," << magnetic_vector_potential[3 * node_index + 2] << "};\n";
        }
    }
    stream << "};\n";
}

template <class Logger>
Result run_on_quadrilateral_grid(
        std::filesystem::path const& output_view_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        sil::onelab_interface::gmsh::QuadrilateralMesh const& mesh,
        Logger&& logger)
{
    SIMILIE_DEBUG_LOG("similie_onelab_linear_magnetostatics_run_on_quadrilateral_grid");
    sil::onelab_interface::gmsh::StructuredGrid2D const grid
            = sil::onelab_interface::gmsh::build_structured_grid(mesh);
    log_info(
            logger,
            "SimiLie structured rectilinear quadrilateral mesh validated ("
                    + std::to_string(grid.ordered_nodes.size()) + " nodes, dimensions="
                    + std::to_string(grid.nx()) + "x" + std::to_string(grid.ny()) + ")");

    Result result;
    result.topology = "quadrilateral";
    result.node_count = grid.ordered_nodes.size();
    result.mesh_dimensions = {grid.nx(), grid.ny(), 1};
    result.num_cells = grid.ncell_x() * grid.ncell_y();

    std::vector<CellInputFields> cell_inputs(result.num_cells);
    for (std::size_t cell_index = 0; cell_index < result.num_cells; ++cell_index) {
        CellInputFields field {
                .mu = inputs.mu0,
                .current_density = {0.0, 0.0, 0.0},
                .nonlinear_material = false,
        };
        int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
        if (has_tag(inputs.magnetic_material_tags, physical_tag)) {
            field.mu = inputs.core_mu;
            field.nonlinear_material = inputs.use_nonlinear_magnetic_material;
            ++result.num_core_cells;
        } else if (has_tag(inputs.positive_electrical_conductor_tags, physical_tag)) {
            field.current_density[2] = inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else if (has_tag(inputs.negative_electrical_conductor_tags, physical_tag)) {
            field.current_density[2] = -inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else {
            ++result.num_air_cells;
        }
        cell_inputs[cell_index] = field;
    }

    Kokkos::View<double*> x_coords("similie_x_coords", grid.nx());
    Kokkos::View<double*> y_coords("similie_y_coords", grid.ny());
    auto x_coords_host = Kokkos::create_mirror_view(x_coords);
    auto y_coords_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < grid.nx(); ++i) {
        x_coords_host(i) = grid.x_coords[i];
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        y_coords_host(j) = grid.y_coords[j];
    }
    Kokkos::deep_copy(x_coords, x_coords_host);
    Kokkos::deep_copy(y_coords, y_coords_host);

    std::size_t const num_nodes = grid.nx() * grid.ny();
    Kokkos::View<double**> rhs("similie_rhs", num_nodes, 1);
    Kokkos::View<double**> magnetic_vector_potential_z_view("similie_Az", num_nodes, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index = grid.node_index(i, j);
            bool const boundary = (i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny());
            if (boundary) {
                rhs_host(node_index, 0) = 0.0;
                continue;
            }
            if (solver_settings.criterion == solvers::Criterion::PotentialTemporalDerivative) {
                rhs_host(node_index, 0) = 0.0;
                continue;
            }
            double accumulated_current_density_z = 0.0;
            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    std::ptrdiff_t const ci = static_cast<std::ptrdiff_t>(i) + di;
                    std::ptrdiff_t const cj = static_cast<std::ptrdiff_t>(j) + dj;
                    if (ci < 0 || cj < 0 || ci >= static_cast<std::ptrdiff_t>(grid.ncell_x())
                        || cj >= static_cast<std::ptrdiff_t>(grid.ncell_y())) {
                        continue;
                    }
                    std::size_t const cell_i = static_cast<std::size_t>(ci);
                    std::size_t const cell_j = static_cast<std::size_t>(cj);
                    double const cell_area = (grid.x_coords[cell_i + 1] - grid.x_coords[cell_i])
                                             * (grid.y_coords[cell_j + 1] - grid.y_coords[cell_j]);
                    accumulated_current_density_z
                            += 0.25 * cell_area
                               * cell_inputs[grid.cell_index(cell_i, cell_j)].current_density[2];
                }
            }
            rhs_host(node_index, 0) = accumulated_current_density_z;
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);
    log_info(logger, "SimiLie right-hand side assembled on rectilinear nodes");

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    magnetostatics_local::scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY,
                    magnetostatics_local::ScalarPotentialIndex>(
                    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(
                            ddc::DiscreteElement<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(0, 0),
                            ddc::DiscreteVector<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(grid.nx(), grid.ny())),
                    sil::tensor::TensorAccessor<magnetostatics_local::ScalarPotentialIndex>()
                            .domain()),
            ddc::KokkosAllocator<double, memory_space>());
    magnetostatics_local::ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    magnetostatics_local::scalar_tensor_alloc_type<memory_space> nonlinear_mask_alloc(
            ddc::DiscreteDomain<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY,
                    magnetostatics_local::ScalarPotentialIndex>(
                    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(
                            ddc::DiscreteElement<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(0, 0),
                            ddc::DiscreteVector<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(grid.nx(), grid.ny())),
                    sil::tensor::TensorAccessor<magnetostatics_local::ScalarPotentialIndex>()
                            .domain()),
            ddc::KokkosAllocator<double, memory_space>());
    magnetostatics_local::ScalarPotentialTensor2D<memory_space> nonlinear_mask_tensor(
            nonlinear_mask_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    auto nonlinear_mask_host
            = Kokkos::create_mirror_view(nonlinear_mask_alloc.allocation_kokkos_view());
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            double accumulated_mu = 0.0;
            double accumulated_nonlinear_mask = 0.0;
            std::size_t count = 0;
            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    std::ptrdiff_t const ci = static_cast<std::ptrdiff_t>(i) + di;
                    std::ptrdiff_t const cj = static_cast<std::ptrdiff_t>(j) + dj;
                    if (ci < 0 || cj < 0 || ci >= static_cast<std::ptrdiff_t>(grid.ncell_x())
                        || cj >= static_cast<std::ptrdiff_t>(grid.ncell_y())) {
                        continue;
                    }
                    CellInputFields const& cell_input = cell_inputs[grid.cell_index(
                            static_cast<std::size_t>(ci),
                            static_cast<std::size_t>(cj))];
                    accumulated_mu += cell_input.mu;
                    accumulated_nonlinear_mask += (cell_input.nonlinear_material ? 1.0 : 0.0);
                    ++count;
                }
            }
            mu_host(i, j, 0)
                    = count == 0 ? inputs.mu0 : accumulated_mu / static_cast<double>(count);
            nonlinear_mask_host(i, j, 0)
                    = count == 0 ? 0.0 : accumulated_nonlinear_mask / static_cast<double>(count);
        }
    }
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);
    Kokkos::deep_copy(nonlinear_mask_alloc.allocation_kokkos_view(), nonlinear_mask_host);

    auto solve_with_equations = [&](auto equations) {
        SIMILIE_DEBUG_LOG("similie_onelab_linear_magnetostatics_build_operator_2d");
        auto const operator_model = magnetostatics_local::MagnetostaticsOperator2D<
                memory_space,
                decltype(equations)>(equations, x_coords, y_coords, solver_settings.criterion);
        log_info(
                logger,
                solver_settings.use_matrix_free
                        ? "SimiLie starting matrix-free preconditioned conjugate-gradient solve"
                        : "SimiLie starting assembled-matrix Ginkgo preconditioned "
                          "conjugate-gradient "
                          "solve");
        result.solver_diagnostics = solvers::minimize_strong_formulation_residual(
                Kokkos::DefaultExecutionSpace(),
                operator_model,
                rhs,
                magnetic_vector_potential_z_view,
                solver_settings);
        log_info(
                logger,
                solver_settings.use_matrix_free
                        ? "SimiLie matrix-free preconditioned conjugate-gradient solve finished"
                        : "SimiLie assembled-matrix Ginkgo preconditioned conjugate-gradient solve "
                          "finished");
    };
    if (inputs.use_nonlinear_magnetic_material) {
        validate_nonlinear_bh_curve(inputs.nonlinear_bh_curve);
        using curve_type = physics::magnetostatics::InterpolatedNonlinearBHCurve<64>;
        auto const nonlinear_bh_curve = curve_type(
                magnetostatics_local::to_padded_std_array<64>(inputs.nonlinear_b_samples),
                magnetostatics_local::to_padded_std_array<64>(inputs.nonlinear_h_samples),
                inputs.nonlinear_b_samples.size());
        auto const equations = magnetostatics_local::MixedMaterialMagnetostaticsEquations(
                mu_tensor,
                nonlinear_mask_tensor,
                nonlinear_bh_curve);
        solve_with_equations(equations);
    } else {
        auto const hamiltonian
                = magnetostatics_local::LinearMagnetostaticsHamiltonian<decltype(mu_tensor)>(
                        mu_tensor);
        solve_with_equations(physics::HamiltonEquations {hamiltonian});
    }

    auto magnetic_vector_potential_z_host = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), magnetic_vector_potential_z_view);
    std::vector<double> magnetic_vector_potential(3 * num_nodes, 0.0);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index = grid.node_index(i, j);
            magnetic_vector_potential[3 * node_index + 2]
                    = magnetic_vector_potential_z_host(node_index, 0);
        }
    }

    for (double value : magnetic_vector_potential) {
        result.max_abs_potential = std::max(result.max_abs_potential, std::abs(value));
    }
    log_info(logger, "SimiLie starting magnetostatics post-processing");

    auto node_value_z = [&](std::size_t i, std::size_t j) {
        return magnetic_vector_potential[3 * grid.node_index(i, j) + 2];
    };

    auto const node_domain = ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY>(
            ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(0, 0),
            ddc::DiscreteVector<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY>(grid.nx(), grid.ny()));
    auto const cell_domain = ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY>(
            ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(0, 0),
            ddc::DiscreteVector<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY>(grid.ncell_x(), grid.ncell_y()));
    std::vector<CellPostProcessFields> cell_outputs(result.num_cells);
    auto fill_cell_outputs = [&](auto const& nonlinear_constitutive_law) {
        fill_post_process_fields_on_cell_domain(
                cell_domain,
                node_domain,
                [&](auto node_elem) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(node_elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(node_elem).uid());
                    return std::array<double, 2> {
                            grid.x_coords[i],
                            grid.y_coords[j],
                    };
                },
                [&](auto elem) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                    return cell_inputs[grid.cell_index(i, j)].mu;
                },
                [&](auto elem) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                    return cell_inputs[grid.cell_index(i, j)].nonlinear_material;
                },
                nonlinear_constitutive_law,
                node_value_z,
                [&](auto elem,
                    std::array<double, 3> const& magnetic_induction,
                    std::array<double, 3> const& magnetic_field) {
                    std::size_t const i = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                    std::size_t const j = static_cast<std::size_t>(
                            ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                    std::size_t const cell_index = grid.cell_index(i, j);
                    cell_outputs[cell_index]
                            = make_cell_post_process_fields(magnetic_induction, magnetic_field);
                });
    };
    if (inputs.use_nonlinear_magnetic_material) {
        auto const nonlinear_constitutive_law
                = physics::magnetostatics::NonlinearMagneticInductionToMagneticField(
                        physics::magnetostatics::InterpolatedNonlinearBHCurve<64>(
                                magnetostatics_local::to_padded_std_array<64>(
                                        inputs.nonlinear_b_samples),
                                magnetostatics_local::to_padded_std_array<64>(
                                        inputs.nonlinear_h_samples),
                                inputs.nonlinear_b_samples.size()));
        fill_cell_outputs(nonlinear_constitutive_law);
    } else {
        auto const dummy_nonlinear_constitutive_law
                = [](auto, auto) { return std::array<double, 3> {0.0, 0.0, 0.0}; };
        fill_cell_outputs(dummy_nonlinear_constitutive_law);
    }
    for (CellPostProcessFields const& cell_output : cell_outputs) {
        for (double value : cell_output.magnetic_induction) {
            result.max_abs_induction = std::max(result.max_abs_induction, std::abs(value));
        }
        for (double value : cell_output.magnetic_field) {
            result.max_abs_field = std::max(result.max_abs_field, std::abs(value));
        }
    }
    fill_force_density_on_quadrilateral_grid(grid, cell_outputs);
    std::vector<DiagnosticFaceSample> diagnostic_face_samples;
    diagnostic_face_samples.reserve(grid.ncell_x());
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            std::size_t const cell_index = grid.cell_index(i, j);
            int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
            double const cell_area = (grid.x_coords[i + 1] - grid.x_coords[i])
                                     * (grid.y_coords[j + 1] - grid.y_coords[j]);
            if (has_tag(inputs.positive_electrical_conductor_tags, physical_tag)) {
                result.diagnostic_current_integral
                        += cell_area * cell_inputs[cell_index].current_density[2];
                ++result.num_current_cells;
            }
            if (!has_tag(inputs.diagnostic_region_tags, physical_tag)) {
                continue;
            }
            bool const has_upper_neighbor = j + 1 < grid.ncell_y();
            bool const upper_neighbor_in_region
                    = has_upper_neighbor
                      && has_tag(
                              inputs.diagnostic_region_tags,
                              grid.ordered_cells[grid.cell_index(i, j + 1)].physical_tag);
            if (upper_neighbor_in_region) {
                continue;
            }
            double const face_measure = (grid.x_coords[i + 1] - grid.x_coords[i]) * inputs.length_z;
            result.diagnostic_flux_integral
                    += face_measure * cell_outputs[cell_index].magnetic_induction[1];
            std::array<double, 3> const traction = traction_on_positive_y_face(
                    cell_outputs[cell_index].magnetic_induction,
                    cell_outputs[cell_index].magnetic_field);
            result.diagnostic_surface_measure += face_measure;
            result.diagnostic_traction_integral[0] += face_measure * traction[0];
            result.diagnostic_traction_integral[1] += face_measure * traction[1];
            result.diagnostic_traction_integral[2] += face_measure * traction[2];
            result.diagnostic_traction_magnitude_integral
                    += face_measure
                       * std::sqrt(
                               traction[0] * traction[0] + traction[1] * traction[1]
                               + traction[2] * traction[2]);
            diagnostic_face_samples.push_back(
                    DiagnosticFaceSample {
                            .x = grid.cell_center_x(i),
                            .z = grid.z_value,
                            .measure = face_measure,
                            .traction = traction,
                    });
            ++result.num_diagnostic_faces;
        }
    }
    apply_x_mirror_symmetry_projection_to_traction_integral(diagnostic_face_samples, result);

    write_results_view(
            output_view_file,
            grid,
            cell_inputs,
            cell_outputs,
            magnetic_vector_potential);
    log_info(logger, "SimiLie magnetostatics post-processing exported");
    return result;
}

inline void write_results_view(
        std::filesystem::path const& output_view_file,
        sil::onelab_interface::gmsh::StructuredGrid3D const& grid,
        std::vector<CellInputFields> const& cell_inputs,
        std::vector<CellPostProcessFields> const& cell_outputs,
        std::vector<double> const& magnetic_vector_potential)
{
    std::ofstream stream(output_view_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_view_file.string());
    }

    stream << "View \"SimiLie linear magnetostatics permeability\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto const& cell_input = cell_inputs[grid.cell_index(i, j, k)];
                stream << "SP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                       << grid.cell_center_z(k) << "){" << cell_input.mu << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics current density\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto const& cell_input = cell_inputs[grid.cell_index(i, j, k)];
                stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                       << grid.cell_center_z(k) << "){" << cell_input.current_density[0] << ","
                       << cell_input.current_density[1] << "," << cell_input.current_density[2]
                       << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic induction\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto const& cell_output = cell_outputs[grid.cell_index(i, j, k)];
                stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                       << grid.cell_center_z(k) << "){" << cell_output.magnetic_induction[0] << ","
                       << cell_output.magnetic_induction[1] << ","
                       << cell_output.magnetic_induction[2] << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic field\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto const& cell_output = cell_outputs[grid.cell_index(i, j, k)];
                stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                       << grid.cell_center_z(k) << "){" << cell_output.magnetic_field[0] << ","
                       << cell_output.magnetic_field[1] << "," << cell_output.magnetic_field[2]
                       << "};\n";
            }
        }
    }
    stream << "};\n";

    auto write_stress_view = [&](std::string_view view_name, std::size_t component) {
        stream << "View \"" << view_name << "\" {\n";
        for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
            for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
                for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                    auto const& cell_output = cell_outputs[grid.cell_index(i, j, k)];
                    stream << "SP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                           << grid.cell_center_z(k) << "){" << cell_output.maxwell_stress[component]
                           << "};\n";
                }
            }
        }
        stream << "};\n";
    };
    write_stress_view("SimiLie linear magnetostatics Maxwell stress xx", 0);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress yy", 1);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress zz", 2);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress xy", 3);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress xz", 4);
    write_stress_view("SimiLie linear magnetostatics Maxwell stress yz", 5);

    stream << "View \"SimiLie linear magnetostatics force density\" {\n";
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                auto const& cell_output = cell_outputs[grid.cell_index(i, j, k)];
                stream << "VP(" << grid.cell_center_x(i) << "," << grid.cell_center_y(j) << ","
                       << grid.cell_center_z(k) << "){" << cell_output.force_density[0] << ","
                       << cell_output.force_density[1] << "," << cell_output.force_density[2]
                       << "};\n";
            }
        }
    }
    stream << "};\n";

    stream << "View \"SimiLie linear magnetostatics magnetic vector potential\" {\n";
    for (std::size_t k = 0; k < grid.nz(); ++k) {
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                std::size_t const node_index = grid.node_index(i, j, k);
                stream << "VP(" << grid.x_coords[i] << "," << grid.y_coords[j] << ","
                       << grid.z_coords[k] << "){0,0,"
                       << magnetic_vector_potential[3 * node_index + 2] << "};\n";
            }
        }
    }
    stream << "};\n";
}

template <class Logger>
Result run_on_hexahedral_grid(
        std::filesystem::path const& output_view_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        sil::onelab_interface::gmsh::HexahedralMesh const& mesh,
        Logger&& logger)
{
    SIMILIE_DEBUG_LOG("similie_onelab_linear_magnetostatics_run_on_hexahedral_grid");
    sil::onelab_interface::gmsh::StructuredGrid3D const grid
            = sil::onelab_interface::gmsh::build_structured_grid(mesh);
    log_info(
            logger,
            "SimiLie structured rectilinear hexahedral mesh validated ("
                    + std::to_string(grid.ordered_nodes.size())
                    + " nodes, dimensions=" + std::to_string(grid.nx()) + "x"
                    + std::to_string(grid.ny()) + "x" + std::to_string(grid.nz()) + ")");

    Result result;
    result.topology = "hexahedral";
    result.node_count = grid.ordered_nodes.size();
    result.mesh_dimensions = {grid.nx(), grid.ny(), grid.nz()};
    result.num_cells = grid.ncell_x() * grid.ncell_y() * grid.ncell_z();

    std::vector<CellInputFields> cell_inputs_3d(result.num_cells);
    for (std::size_t cell_index = 0; cell_index < result.num_cells; ++cell_index) {
        CellInputFields field {
                .mu = inputs.mu0,
                .current_density = {0.0, 0.0, 0.0},
                .nonlinear_material = false,
        };
        int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
        if (has_tag(inputs.magnetic_material_tags, physical_tag)) {
            field.mu = inputs.core_mu;
            field.nonlinear_material = inputs.use_nonlinear_magnetic_material;
            ++result.num_core_cells;
        } else if (has_tag(inputs.positive_electrical_conductor_tags, physical_tag)) {
            field.current_density[2] = inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else if (has_tag(inputs.negative_electrical_conductor_tags, physical_tag)) {
            field.current_density[2] = -inputs.current_density_magnitude;
            ++result.num_coil_cells;
        } else {
            ++result.num_air_cells;
        }
        cell_inputs_3d[cell_index] = field;
    }

    std::size_t const num_cells_xy = grid.ncell_x() * grid.ncell_y();
    std::vector<CellInputFields> cell_inputs_2d(num_cells_xy);
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            CellInputFields averaged {
                    .mu = 0.0,
                    .current_density = {0.0, 0.0, 0.0},
                    .nonlinear_material = false,
            };
            std::size_t nonlinear_count = 0;
            for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
                CellInputFields const& current = cell_inputs_3d[grid.cell_index(i, j, k)];
                averaged.mu += current.mu;
                averaged.current_density[0] += current.current_density[0];
                averaged.current_density[1] += current.current_density[1];
                averaged.current_density[2] += current.current_density[2];
                nonlinear_count += current.nonlinear_material ? 1U : 0U;
            }
            double const scale = 1.0 / static_cast<double>(grid.ncell_z());
            averaged.mu *= scale;
            averaged.current_density[0] *= scale;
            averaged.current_density[1] *= scale;
            averaged.current_density[2] *= scale;
            averaged.nonlinear_material = 2 * nonlinear_count >= grid.ncell_z();
            cell_inputs_2d[i + grid.ncell_x() * j] = averaged;
        }
    }

    Kokkos::View<double*> x_coords("similie_x_coords", grid.nx());
    Kokkos::View<double*> y_coords("similie_y_coords", grid.ny());
    auto x_coords_host = Kokkos::create_mirror_view(x_coords);
    auto y_coords_host = Kokkos::create_mirror_view(y_coords);
    for (std::size_t i = 0; i < grid.nx(); ++i) {
        x_coords_host(i) = grid.x_coords[i];
    }
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        y_coords_host(j) = grid.y_coords[j];
    }
    Kokkos::deep_copy(x_coords, x_coords_host);
    Kokkos::deep_copy(y_coords, y_coords_host);

    std::size_t const num_nodes_xy = grid.nx() * grid.ny();
    Kokkos::View<double**> rhs("similie_rhs", num_nodes_xy, 1);
    Kokkos::View<double**> magnetic_vector_potential_z_view("similie_Az", num_nodes_xy, 1);
    auto rhs_host = Kokkos::create_mirror_view(rhs);
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            std::size_t const node_index = i + grid.nx() * j;
            bool const boundary = (i == 0 || j == 0 || i + 1 == grid.nx() || j + 1 == grid.ny());
            if (boundary) {
                rhs_host(node_index, 0) = 0.0;
                continue;
            }
            if (solver_settings.criterion == solvers::Criterion::PotentialTemporalDerivative) {
                rhs_host(node_index, 0) = 0.0;
                continue;
            }
            double accumulated_current_density_z = 0.0;
            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    std::ptrdiff_t const ci = static_cast<std::ptrdiff_t>(i) + di;
                    std::ptrdiff_t const cj = static_cast<std::ptrdiff_t>(j) + dj;
                    if (ci < 0 || cj < 0 || ci >= static_cast<std::ptrdiff_t>(grid.ncell_x())
                        || cj >= static_cast<std::ptrdiff_t>(grid.ncell_y())) {
                        continue;
                    }
                    std::size_t const cell_i = static_cast<std::size_t>(ci);
                    std::size_t const cell_j = static_cast<std::size_t>(cj);
                    double const cell_area = (grid.x_coords[cell_i + 1] - grid.x_coords[cell_i])
                                             * (grid.y_coords[cell_j + 1] - grid.y_coords[cell_j]);
                    accumulated_current_density_z
                            += 0.25 * cell_area
                               * cell_inputs_2d[cell_i + grid.ncell_x() * cell_j]
                                         .current_density[2];
                }
            }
            rhs_host(node_index, 0) = accumulated_current_density_z;
        }
    }
    Kokkos::deep_copy(rhs, rhs_host);
    log_info(logger, "SimiLie right-hand side assembled on rectilinear nodes");

    using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
    magnetostatics_local::scalar_tensor_alloc_type<memory_space> mu_alloc(
            ddc::DiscreteDomain<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY,
                    magnetostatics_local::ScalarPotentialIndex>(
                    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(
                            ddc::DiscreteElement<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(0, 0),
                            ddc::DiscreteVector<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(grid.nx(), grid.ny())),
                    sil::tensor::TensorAccessor<magnetostatics_local::ScalarPotentialIndex>()
                            .domain()),
            ddc::KokkosAllocator<double, memory_space>());
    magnetostatics_local::ScalarPotentialTensor2D<memory_space> mu_tensor(mu_alloc);
    magnetostatics_local::scalar_tensor_alloc_type<memory_space> nonlinear_mask_alloc(
            ddc::DiscreteDomain<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY,
                    magnetostatics_local::ScalarPotentialIndex>(
                    ddc::DiscreteDomain<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(
                            ddc::DiscreteElement<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(0, 0),
                            ddc::DiscreteVector<
                                    magnetostatics_local::DDimX,
                                    magnetostatics_local::DDimY>(grid.nx(), grid.ny())),
                    sil::tensor::TensorAccessor<magnetostatics_local::ScalarPotentialIndex>()
                            .domain()),
            ddc::KokkosAllocator<double, memory_space>());
    magnetostatics_local::ScalarPotentialTensor2D<memory_space> nonlinear_mask_tensor(
            nonlinear_mask_alloc);
    auto mu_host = Kokkos::create_mirror_view(mu_alloc.allocation_kokkos_view());
    auto nonlinear_mask_host
            = Kokkos::create_mirror_view(nonlinear_mask_alloc.allocation_kokkos_view());
    for (std::size_t j = 0; j < grid.ny(); ++j) {
        for (std::size_t i = 0; i < grid.nx(); ++i) {
            double accumulated_mu = 0.0;
            double accumulated_nonlinear_mask = 0.0;
            std::size_t count = 0;
            for (int dj = -1; dj <= 0; ++dj) {
                for (int di = -1; di <= 0; ++di) {
                    std::ptrdiff_t const ci = static_cast<std::ptrdiff_t>(i) + di;
                    std::ptrdiff_t const cj = static_cast<std::ptrdiff_t>(j) + dj;
                    if (ci < 0 || cj < 0 || ci >= static_cast<std::ptrdiff_t>(grid.ncell_x())
                        || cj >= static_cast<std::ptrdiff_t>(grid.ncell_y())) {
                        continue;
                    }
                    CellInputFields const& cell_input = cell_inputs_2d
                            [static_cast<std::size_t>(ci)
                             + grid.ncell_x() * static_cast<std::size_t>(cj)];
                    accumulated_mu += cell_input.mu;
                    accumulated_nonlinear_mask += (cell_input.nonlinear_material ? 1.0 : 0.0);
                    ++count;
                }
            }
            mu_host(i, j, 0)
                    = count == 0 ? inputs.mu0 : accumulated_mu / static_cast<double>(count);
            nonlinear_mask_host(i, j, 0)
                    = count == 0 ? 0.0 : accumulated_nonlinear_mask / static_cast<double>(count);
        }
    }
    Kokkos::deep_copy(mu_alloc.allocation_kokkos_view(), mu_host);
    Kokkos::deep_copy(nonlinear_mask_alloc.allocation_kokkos_view(), nonlinear_mask_host);

    auto solve_with_equations = [&](auto equations) {
        SIMILIE_DEBUG_LOG("similie_onelab_linear_magnetostatics_build_operator_3d_reduced_to_2d");
        auto const operator_model = magnetostatics_local::MagnetostaticsOperator2D<
                memory_space,
                decltype(equations)>(equations, x_coords, y_coords, solver_settings.criterion);
        log_info(
                logger,
                solver_settings.use_matrix_free
                        ? "SimiLie starting matrix-free preconditioned conjugate-gradient solve"
                        : "SimiLie starting assembled-matrix Ginkgo preconditioned "
                          "conjugate-gradient "
                          "solve");
        result.solver_diagnostics = solvers::minimize_strong_formulation_residual(
                Kokkos::DefaultExecutionSpace(),
                operator_model,
                rhs,
                magnetic_vector_potential_z_view,
                solver_settings);
        log_info(
                logger,
                solver_settings.use_matrix_free
                        ? "SimiLie matrix-free preconditioned conjugate-gradient solve finished"
                        : "SimiLie assembled-matrix Ginkgo preconditioned conjugate-gradient solve "
                          "finished");
    };
    if (inputs.use_nonlinear_magnetic_material) {
        validate_nonlinear_bh_curve(inputs.nonlinear_bh_curve);
        using curve_type = physics::magnetostatics::InterpolatedNonlinearBHCurve<64>;
        auto const nonlinear_bh_curve = curve_type(
                magnetostatics_local::to_padded_std_array<64>(inputs.nonlinear_b_samples),
                magnetostatics_local::to_padded_std_array<64>(inputs.nonlinear_h_samples),
                inputs.nonlinear_b_samples.size());
        auto const equations = magnetostatics_local::MixedMaterialMagnetostaticsEquations(
                mu_tensor,
                nonlinear_mask_tensor,
                nonlinear_bh_curve);
        solve_with_equations(equations);
    } else {
        auto const hamiltonian
                = magnetostatics_local::LinearMagnetostaticsHamiltonian<decltype(mu_tensor)>(
                        mu_tensor);
        solve_with_equations(physics::HamiltonEquations {hamiltonian});
    }

    auto magnetic_vector_potential_z_host = Kokkos::
            create_mirror_view_and_copy(Kokkos::HostSpace(), magnetic_vector_potential_z_view);
    std::vector<double> magnetic_vector_potential(3 * grid.ordered_nodes.size(), 0.0);
    for (std::size_t k = 0; k < grid.nz(); ++k) {
        for (std::size_t j = 0; j < grid.ny(); ++j) {
            for (std::size_t i = 0; i < grid.nx(); ++i) {
                std::size_t const node_index_2d = i + grid.nx() * j;
                std::size_t const node_index_3d = grid.node_index(i, j, k);
                magnetic_vector_potential[3 * node_index_3d + 2]
                        = magnetic_vector_potential_z_host(node_index_2d, 0);
            }
        }
    }

    for (double value : magnetic_vector_potential) {
        result.max_abs_potential = std::max(result.max_abs_potential, std::abs(value));
    }
    log_info(logger, "SimiLie starting magnetostatics post-processing");

    auto const node_domain = ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY>(
            ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(0, 0),
            ddc::DiscreteVector<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY>(grid.nx(), grid.ny()));
    auto const cell_domain = ddc::DiscreteDomain<
            magnetostatics_local::DDimX,
            magnetostatics_local::DDimY>(
            ddc::DiscreteElement<magnetostatics_local::DDimX, magnetostatics_local::DDimY>(0, 0),
            ddc::DiscreteVector<
                    magnetostatics_local::DDimX,
                    magnetostatics_local::DDimY>(grid.ncell_x(), grid.ncell_y()));
    std::vector<CellPostProcessFields> cell_outputs(result.num_cells);
    auto fill_slice_outputs = [&](auto const& nonlinear_constitutive_law) {
        for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
            auto node_value_z = [&](std::size_t node_i, std::size_t node_j) {
                return magnetic_vector_potential[3 * grid.node_index(node_i, node_j, k) + 2];
            };
            fill_post_process_fields_on_cell_domain(
                    cell_domain,
                    node_domain,
                    [&](auto node_elem) {
                        std::size_t const i = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimX>(node_elem).uid());
                        std::size_t const j = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimY>(node_elem).uid());
                        return std::array<double, 2> {
                                grid.x_coords[i],
                                grid.y_coords[j],
                        };
                    },
                    [&](auto elem) {
                        std::size_t const i = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                        std::size_t const j = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                        return cell_inputs_3d[grid.cell_index(i, j, k)].mu;
                    },
                    [&](auto elem) {
                        std::size_t const i = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                        std::size_t const j = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                        return cell_inputs_3d[grid.cell_index(i, j, k)].nonlinear_material;
                    },
                    nonlinear_constitutive_law,
                    node_value_z,
                    [&](auto elem,
                        std::array<double, 3> const& magnetic_induction,
                        std::array<double, 3> const& magnetic_field) {
                        std::size_t const i = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimX>(elem).uid());
                        std::size_t const j = static_cast<std::size_t>(
                                ddc::DiscreteElement<magnetostatics_local::DDimY>(elem).uid());
                        std::size_t const cell_index = grid.cell_index(i, j, k);
                        cell_outputs[cell_index]
                                = make_cell_post_process_fields(magnetic_induction, magnetic_field);
                    });
        }
    };
    if (inputs.use_nonlinear_magnetic_material) {
        auto const nonlinear_constitutive_law
                = physics::magnetostatics::NonlinearMagneticInductionToMagneticField(
                        physics::magnetostatics::InterpolatedNonlinearBHCurve<64>(
                                magnetostatics_local::to_padded_std_array<64>(
                                        inputs.nonlinear_b_samples),
                                magnetostatics_local::to_padded_std_array<64>(
                                        inputs.nonlinear_h_samples),
                                inputs.nonlinear_b_samples.size()));
        fill_slice_outputs(nonlinear_constitutive_law);
    } else {
        auto const dummy_nonlinear_constitutive_law
                = [](auto, auto) { return std::array<double, 3> {0.0, 0.0, 0.0}; };
        fill_slice_outputs(dummy_nonlinear_constitutive_law);
    }
    for (CellPostProcessFields const& cell_output : cell_outputs) {
        for (double value : cell_output.magnetic_induction) {
            result.max_abs_induction = std::max(result.max_abs_induction, std::abs(value));
        }
        for (double value : cell_output.magnetic_field) {
            result.max_abs_field = std::max(result.max_abs_field, std::abs(value));
        }
    }
    fill_force_density_on_hexahedral_grid_xy_slices(grid, cell_outputs);
    std::vector<DiagnosticFaceSample> diagnostic_face_samples;
    diagnostic_face_samples.reserve(grid.ncell_x() * grid.ncell_z());
    for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
        for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
            std::size_t const cell_index_xy = i + grid.ncell_x() * j;
            std::size_t const cell_index_3d = grid.cell_index(i, j, 0);
            int const physical_tag = grid.ordered_cells[cell_index_3d].physical_tag;
            if (!has_tag(inputs.positive_electrical_conductor_tags, physical_tag)) {
                continue;
            }
            double const cell_cross_section_area = (grid.x_coords[i + 1] - grid.x_coords[i])
                                                   * (grid.y_coords[j + 1] - grid.y_coords[j]);
            result.diagnostic_current_integral
                    += cell_cross_section_area * cell_inputs_2d[cell_index_xy].current_density[2];
            ++result.num_current_cells;
        }
    }
    for (std::size_t k = 0; k < grid.ncell_z(); ++k) {
        for (std::size_t j = 0; j < grid.ncell_y(); ++j) {
            for (std::size_t i = 0; i < grid.ncell_x(); ++i) {
                std::size_t const cell_index = grid.cell_index(i, j, k);
                int const physical_tag = grid.ordered_cells[cell_index].physical_tag;
                double const cell_volume = (grid.x_coords[i + 1] - grid.x_coords[i])
                                           * (grid.y_coords[j + 1] - grid.y_coords[j])
                                           * (grid.z_coords[k + 1] - grid.z_coords[k]);
                if (!has_tag(inputs.diagnostic_region_tags, physical_tag)) {
                    continue;
                }
                bool const has_upper_neighbor = j + 1 < grid.ncell_y();
                bool const upper_neighbor_in_region
                        = has_upper_neighbor
                          && has_tag(
                                  inputs.diagnostic_region_tags,
                                  grid.ordered_cells[grid.cell_index(i, j + 1, k)].physical_tag);
                if (upper_neighbor_in_region) {
                    continue;
                }
                double const face_measure = (grid.x_coords[i + 1] - grid.x_coords[i])
                                            * (grid.z_coords[k + 1] - grid.z_coords[k]);
                result.diagnostic_flux_integral
                        += face_measure * cell_outputs[cell_index].magnetic_induction[1];
                std::array<double, 3> const traction = traction_on_positive_y_face(
                        cell_outputs[cell_index].magnetic_induction,
                        cell_outputs[cell_index].magnetic_field);
                result.diagnostic_surface_measure += face_measure;
                result.diagnostic_traction_integral[0] += face_measure * traction[0];
                result.diagnostic_traction_integral[1] += face_measure * traction[1];
                result.diagnostic_traction_integral[2] += face_measure * traction[2];
                result.diagnostic_traction_magnitude_integral
                        += face_measure
                           * std::sqrt(
                                   traction[0] * traction[0] + traction[1] * traction[1]
                                   + traction[2] * traction[2]);
                diagnostic_face_samples.push_back(
                        DiagnosticFaceSample {
                                .x = grid.cell_center_x(i),
                                .z = grid.cell_center_z(k),
                                .measure = face_measure,
                                .traction = traction,
                        });
                ++result.num_diagnostic_faces;
            }
        }
    }
    apply_x_mirror_symmetry_projection_to_traction_integral(diagnostic_face_samples, result);

    write_results_view(
            output_view_file,
            grid,
            cell_inputs_3d,
            cell_outputs,
            magnetic_vector_potential);
    log_info(logger, "SimiLie magnetostatics post-processing exported");
    return result;
}

} // namespace detail

template <class Logger>
Result run(
        std::filesystem::path const& mesh_file,
        std::filesystem::path const& output_view_file,
        Inputs const& inputs,
        solvers::StrongFormulationSolverSettings const& solver_settings,
        Logger&& logger)
{
    auto const mesh = sil::onelab_interface::gmsh::parse_supported_msh2_mesh(mesh_file);
    if (std::holds_alternative<sil::onelab_interface::gmsh::QuadrilateralMesh>(mesh)) {
        return detail::run_on_quadrilateral_grid(
                output_view_file,
                inputs,
                solver_settings,
                std::get<sil::onelab_interface::gmsh::QuadrilateralMesh>(mesh),
                std::forward<Logger>(logger));
    }
    return detail::run_on_hexahedral_grid(
            output_view_file,
            inputs,
            solver_settings,
            std::get<sil::onelab_interface::gmsh::HexahedralMesh>(mesh),
            std::forward<Logger>(logger));
}

} // namespace similie::onelab_interface::magnetostatics_onelab
