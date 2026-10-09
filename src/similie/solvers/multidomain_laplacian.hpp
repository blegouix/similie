// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <similie/exterior/laplacian.hpp>
#include <similie/multidomains/multidomains.hpp>

#include "minimize_strong_formulation_residual.hpp"

namespace similie::solvers {

/** Index-only scalar cochain; sampling maps local nodes to global unknowns.
 * \important This operator and documentation is fully AI-generated.
 * The caller owns the index array in the selected execution space. Nodes use
 * disjoint ownership across domains, in DDC LayoutRight order.
 */
template <class... DDim>
struct IndexedScalarField
{
    using component_type = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using non_indices_domain_t = ddc::DiscreteDomain<DDim...>;
    using discrete_domain_type = ddc::DiscreteDomain<DDim..., component_type>;
    using indices_domain_t = ddc::DiscreteDomain<component_type>;

    non_indices_domain_t grid;
    std::size_t const* indices;

    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return grid;
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION std::size_t global_index(Element elem, Component) const
    {
        std::size_t local = 0;
        ((local = local * grid.template extent<DDim>() + elem.template uid<DDim>()
                  - grid.front().template uid<DDim>()),
         ...);
        assert(sil::misc::domain_contains(grid, elem));
        return indices[local];
    }
};


/** Sparse scalar operator shared by assembled and matrix-free Ginkgo solves.
 * \important This operator and documentation is fully AI-generated.
 * Stores the device CSR arrays assembled from DEC stencils.
 * apply() and create_matrix() use the same coefficients;
 * assemble_matrix_data() exports them to the host only for diagnostics.
 * Nodal row constraints may break symmetry, so no symmetry assumption is made.
 */
class ScalarLaplacianMatrix
{
    Kokkos::View<int*> m_offsets;
    Kokkos::View<int*> m_columns;
    Kokkos::View<double*> m_values;
    std::size_t m_size;

public:
    static constexpr bool IS_LINEAR = true;
    static constexpr bool IS_SYMMETRIC = false;

    /** Adopt CSR arrays assembled in the default memory space, without host packing. */
    ScalarLaplacianMatrix(
            Kokkos::View<int*> offsets,
            Kokkos::View<int*> columns,
            Kokkos::View<double*> values)
        : m_offsets(offsets)
        , m_columns(columns)
        , m_values(values)
        , m_size(offsets.extent(0) - 1)
    {
        if (offsets.extent(0) == 0 || columns.extent(0) != values.extent(0))
            throw std::invalid_argument("inconsistent CSR array extents");
    }

    /** Transfer CSR directly to the solver executor; no host matrix_data is built. */
    std::shared_ptr<gko::matrix::Csr<double, gko::int32>> create_matrix(
            std::shared_ptr<gko::Executor const> const& executor) const
    {
        auto const source = gko::ext::kokkos::create_executor(Kokkos::DefaultExecutionSpace {});
        gko::array<double> values(executor, m_values.extent(0));
        gko::array<gko::int32> columns(executor, m_columns.extent(0)),
                offsets(executor, m_offsets.extent(0));
        executor->copy_from(source.get(), m_values.extent(0), m_values.data(), values.get_data());
        executor->copy_from(
                source.get(),
                m_columns.extent(0),
                m_columns.data(),
                columns.get_data());
        executor->copy_from(
                source.get(),
                m_offsets.extent(0),
                m_offsets.data(),
                offsets.get_data());
        std::shared_ptr<gko::matrix::Csr<double, gko::int32>> matrix(
                gko::matrix::Csr<double, gko::int32>::
                        create(executor,
                               gko::dim<2>(m_size, m_size),
                               std::move(values),
                               std::move(columns),
                               std::move(offsets))
                                .release());
        matrix->sort_by_column_index();
        return matrix;
    }

    [[nodiscard]] std::size_t size() const
    {
        return m_size;
    }

    template <class ExecSpace, class InputView, class OutputView>
    void apply(ExecSpace exec_space, InputView input, OutputView output) const
    {
        auto const offsets = m_offsets;
        auto const columns = m_columns;
        auto const values = m_values;
        Kokkos::parallel_for(
                "similie_sparse_operator",
                Kokkos::RangePolicy<ExecSpace>(exec_space, 0, m_size),
                KOKKOS_LAMBDA(std::size_t row) {
                    double sum = 0.0;
                    for (int slot = offsets(row); slot < offsets(row + 1); ++slot)
                        sum += values(slot) * input(columns(slot), 0);
                    output(row, 0) = sum;
                });
        exec_space.fence();
    }

    friend gko::matrix_data<double, gko::int32> assemble_matrix_data(
            ScalarLaplacianMatrix const& matrix)
    {
        gko::matrix_data<double, gko::int32> data(gko::dim<2>(matrix.m_size, matrix.m_size));
        auto const offsets
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_offsets);
        auto const columns
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_columns);
        auto const values
                = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix.m_values);
        for (std::size_t row = 0; row < matrix.m_size; ++row)
            for (int slot = offsets(row); slot < offsets(row + 1); ++slot)
                data.nonzeros
                        .emplace_back(static_cast<gko::int32>(row), columns(slot), values(slot));
        return data;
    }
};

template <class RhsView>
    requires Kokkos::is_view<RhsView>::value
Kokkos::View<double**> solve_linear_system(
        ScalarLaplacianMatrix const& matrix,
        RhsView rhs,
        StrongFormulationSolverSettings const& settings,
        StrongFormulationSolverDiagnostics& diagnostics)
{
    if (rhs.extent(0) != matrix.size() || rhs.extent(1) != 1)
        throw std::invalid_argument("linear system RHS shape does not match operator");
    Kokkos::View<double**> solution("similie_sparse_solution", matrix.size(), 1);
    diagnostics = minimize_strong_formulation_residual(
            Kokkos::DefaultExecutionSpace {},
            matrix,
            rhs,
            solution,
            settings);
    if (!std::isfinite(diagnostics.final_relative_residual)
        || diagnostics.final_relative_residual > settings.relative_tolerance * 10.0)
        throw std::runtime_error("scalar linear solve did not converge");
    return solution;
}

struct ScalarLinearSystem
{
    ScalarLaplacianMatrix matrix;
    Kokkos::View<double**> rhs;
};
/** Bind lazy *d(phi) cochains, so dual topology rules enter the donor's DEC flux.
 * \important This operator and documentation is fully AI-generated.
 * Every physical node retains its primal field, Cartesian geometry and physics.
 * Boundary nodes retain their policies. Direction is the covariant natural index
 * of the logical axes. Material coefficients multiply the constitutive flux.
 */
template <class Direction, class Domains, class Diffusivity>
auto make_laplacian_flux_domains(
        Domains const& domains,
        Diffusivity diffusivity,
        double jump_scale = 1.0)
{
    return domains.transform_data([&](auto const& data) {
        using Id = typename std::remove_cvref_t<decltype(data)>::id;
        if constexpr (Domains::topology_type::template node<Id>::IS_BOUNDARY) {
            return data;
        } else {
            double const coefficient = diffusivity(data.physics);
            if (!(coefficient > 0.0) || !std::isfinite(coefficient))
                throw std::invalid_argument("scalar diffusivity must be finite and positive");
            auto const field = domains.template field<Id>();
            auto const primal = domains.template extrapolation_rules<Id>(jump_scale);
            auto const geometry = domains.template extrapolation_rules<
                    Id,
                    sil::multidomains::FieldRole::Geometry>();
            auto const boundary_flux = domains.template extrapolation_rules<
                    Id,
                    sil::multidomains::FieldRole::Flux>();
            auto const flux = sil::exterior::make_scalar_laplacian_flux<
                    Direction>(field, data.geometry, primal, geometry, boundary_flux, coefficient);
            return sil::multidomains::domain_data<
                    Id>(data.field, data.geometry, data.physics, flux);
        }
    });
}

namespace detail {
/** Device-local basis probes; no shared probe state or device allocation per row. */
template <std::size_t Capacity>
struct LocalAffineProbe
{
    std::size_t indices[Capacity] {};
    std::size_t count = 0;
    std::size_t active = std::numeric_limits<std::size_t>::max();
    bool recording = false;
    bool overflow = false;

    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double sample(Field const& field, Element element, Component component)
    {
        std::size_t const index = field.global_index(element, component);
        if (recording) {
            bool found = false;
            for (std::size_t i = 0; i < count; ++i)
                found = found || indices[i] == index;
            if (!found) {
                if (count == Capacity)
                    overflow = true;
                else
                    indices[count++] = index;
            }
        }
        return index == active ? 1.0 : 0.0;
    }
};

template <std::size_t Capacity>
struct LocalAffineSampler
{
    LocalAffineProbe<Capacity>* probe;
    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Field const& field, Element element, Component component)
            const
    {
        return probe->sample(field, element, component);
    }
};

template <std::size_t Capacity>
struct AffineStencil
{
    double constant = 0.0;
    std::size_t count = 0;
    std::size_t indices[Capacity] {};
    double weights[Capacity] {};
};


template <std::size_t Capacity, class Evaluate>
KOKKOS_FUNCTION AffineStencil<Capacity> sample_affine(Evaluate evaluate, int& error)
{
    LocalAffineProbe<Capacity> probe;
    LocalAffineSampler<Capacity> const sampler {&probe};
    AffineStencil<Capacity> result;
    probe.recording = true;
    result.constant = evaluate(sampler);
    probe.recording = false;
    if (probe.overflow) {
        error = 2;
        return result;
    }
    for (std::size_t i = 0; i < probe.count; ++i) {
        probe.active = probe.indices[i];
        double const weight = evaluate(sampler) - result.constant;
        if (weight != 0.0) {
            result.indices[result.count] = probe.active;
            result.weights[result.count++] = weight;
        }
    }
    return result;
}

// Each domain writes its owned rows once. Only shared boundary constraints need
// atomics; no sparse hash map, insertion retries or cross-stream row sums.
template <std::size_t Capacity>
struct RowAssembly
{
    Kokkos::View<AffineStencil<Capacity>*> rows;
    Kokkos::View<double*> lower, upper;
    Kokkos::View<int> errors;

    explicit RowAssembly(std::size_t size)
        : rows("dec_stencils", size)
        , lower("dirichlet_lower", size)
        , upper("dirichlet_upper", size)
        , errors("dec_errors")
    {
        if (size == 0 || size > std::size_t(std::numeric_limits<int>::max()) / Capacity)
            throw std::invalid_argument("scalar DEC matrix exceeds 32-bit CSR capacity");
        Kokkos::deep_copy(lower, std::numeric_limits<double>::infinity());
        Kokkos::deep_copy(upper, -std::numeric_limits<double>::infinity());
    }

    KOKKOS_FUNCTION void fail(int code) const
    {
        Kokkos::atomic_fetch_or(&errors(), code);
    }

    KOKKOS_FUNCTION void prescribe(AffineStencil<Capacity> const& sample, double value) const
    {
        if (sample.count != 1 || sample.indices[0] >= rows.extent(0)) {
            fail(8);
            return;
        }
        double const fixed = (value - sample.constant) / sample.weights[0];
        if (!Kokkos::isfinite(fixed)) {
            fail(8);
            return;
        }
        Kokkos::atomic_min(&lower(sample.indices[0]), fixed);
        Kokkos::atomic_max(&upper(sample.indices[0]), fixed);
    }

    void check() const
    {
        int error = 0;
        Kokkos::deep_copy(error, errors);
        if (error & 2)
            throw std::runtime_error("scalar DEC stencil exceeds ProbeCapacity");
        if (error & 8)
            throw std::runtime_error("incompatible or invalid Dirichlet constraints");
        if (error)
            throw std::runtime_error("invalid or empty scalar DEC row");
    }

    ScalarLinearSystem finalize(Kokkos::DefaultExecutionSpace const& exec, bool normalize) const
    {
        check();
        RowAssembly const data = *this;
        Kokkos::View<int*> offsets("dec_offsets", rows.extent(0) + 1);
        Kokkos::View<double**> rhs("dec_rhs", rows.extent(0), 1);
        Kokkos::parallel_for(
                "finish_dec_rows",
                Kokkos::RangePolicy(exec, 0, rows.extent(0)),
                KOKKOS_LAMBDA(std::size_t row) {
                    auto& stencil = data.rows(row);
                    if (Kokkos::isfinite(data.lower(row))) {
                        if (Kokkos::abs(data.upper(row) - data.lower(row))
                            > 1.e-12 * (1.0 + Kokkos::abs(data.lower(row))))
                            data.fail(8);
                        stencil.count = 1;
                        stencil.indices[0] = row;
                        stencil.weights[0] = 1.0;
                        stencil.constant = -data.lower(row);
                    }
                    double scale = 0.0;
                    for (std::size_t i = 0; i < stencil.count; ++i) {
                        if (stencil.indices[i] >= data.rows.extent(0)
                            || !Kokkos::isfinite(stencil.weights[i]))
                            data.fail(4);
                        scale = Kokkos::max(scale, Kokkos::abs(stencil.weights[i]));
                    }
                    if (!(scale > 0.0) || !Kokkos::isfinite(stencil.constant)) {
                        data.fail(4);
                        return;
                    }
                    if (normalize) {
                        for (std::size_t i = 0; i < stencil.count; ++i)
                            stencil.weights[i] /= scale;
                        stencil.constant /= scale;
                    }
                    rhs(row, 0) = -stencil.constant;
                });
        int total = 0;
        Kokkos::parallel_scan(
                "scan_dec_rows",
                Kokkos::RangePolicy(exec, 0, rows.extent(0) + 1),
                KOKKOS_LAMBDA(std::size_t row, int& sum, bool final) {
                    if (final)
                        offsets(row) = sum;
                    if (row < data.rows.extent(0))
                        sum += static_cast<int>(data.rows(row).count);
                },
                total);
        check();
        Kokkos::View<int*> columns("dec_columns", total);
        Kokkos::View<double*> values("dec_values", total);
        Kokkos::parallel_for(
                "pack_dec_rows",
                Kokkos::RangePolicy(exec, 0, rows.extent(0)),
                KOKKOS_LAMBDA(std::size_t row) {
                    for (std::size_t i = 0; i < data.rows(row).count; ++i) {
                        columns(offsets(row) + i) = static_cast<int>(data.rows(row).indices[i]);
                        values(offsets(row) + i) = data.rows(row).weights[i];
                    }
                });
        exec.fence("scalar DEC matrix ready");
        return {ScalarLaplacianMatrix(offsets, columns, values), rhs};
    }
};
template <std::size_t Axis, bool Upper, std::size_t Capacity, class Field, class Primal>
void prescribe_face(
        Kokkos::DefaultExecutionSpace const& exec,
        RowAssembly<Capacity> system,
        Field field,
        Primal primal)
{
    using Rule = std::remove_cvref_t<decltype(primal.template boundary_rule<Axis, Upper>())>;
    if constexpr (requires { Rule::IS_DIRICHLET; }) {
        if constexpr (Rule::IS_DIRICHLET) {
            auto nodes = field.non_indices_domain();
            auto front = nodes.front();
            auto extent = nodes.extents();
            ddc::detail::array(front)[Axis]
                    = ddc::detail::array(Upper ? nodes.back() : nodes.front())[Axis];
            ddc::detail::array(extent)[Axis] = 1;
            // The final endpoint of a physical face can belong to a donor domain.
            [&]<std::size_t... Tangent>(std::index_sequence<Tangent...>) {
                (
                        [&] {
                            using End = std::remove_cvref_t<
                                    decltype(primal.template boundary_rule<Tangent, true>())>;
                            if constexpr (
                                    Tangent != Axis
                                    && sil::exterior::detail::is_interface_rule_v<End>)
                                if (!primal.template boundary_rule<Tangent, true>()
                                             .boundary_map.shared_boundary)
                                    ++ddc::detail::array(extent)[Tangent];
                        }(),
                        ...);
            }(std::make_index_sequence<ddc::type_seq_size_v<
                      ddc::to_type_seq_t<typename Field::non_indices_domain_t>>> {});
            nodes = typename Field::non_indices_domain_t(front, extent);
            ddc::parallel_for_each(
                    "prescribe_dec_face",
                    exec,
                    nodes,
                    KOKKOS_LAMBDA(
                            typename Field::non_indices_domain_t::discrete_element_type elem) {
                        int error = 0;
                        auto const sample = sample_affine<Capacity>(
                                [&](auto sampler) {
                                    return primal
                                            .value(field,
                                                   sampler,
                                                   elem,
                                                   ddc::DiscreteElement<
                                                           typename Field::component_type>(0));
                                },
                                error);
                        auto exterior = elem;
                        ddc::detail::array(exterior)[Axis] += Upper ? 1 : -1;
                        double const value = sil::exterior::detail::EvaluateRuleValue<
                                sil::exterior::detail::
                                        ZeroScalarSampler> {sil::exterior::detail::
                                                                    ZeroScalarSampler {}}(
                                primal.template boundary_rule<Axis, Upper>(),
                                field,
                                exterior,
                                ddc::DiscreteElement<typename Field::component_type>(0));
                        if (error)
                            system.fail(error);
                        else
                            system.prescribe(sample, value);
                    });
        }
    }
}

template <std::size_t Capacity, class Field, class Primal, std::size_t... Axis>
void prescribe_faces(
        Kokkos::DefaultExecutionSpace const& exec,
        RowAssembly<Capacity> system,
        Field field,
        Primal primal,
        std::index_sequence<Axis...>)
{
    (prescribe_face<Axis, false, Capacity>(exec, system, field, primal), ...);
    (prescribe_face<Axis, true, Capacity>(exec, system, field, primal), ...);
}

template <std::size_t Capacity, class Field, class Operator>
void assemble_dec_rows(
        Kokkos::DefaultExecutionSpace const& exec,
        RowAssembly<Capacity> system,
        Field field,
        Operator op)
{
    ddc::parallel_for_each(
            "assemble_dec_laplacian_rows",
            exec,
            field.non_indices_domain(),
            KOKKOS_LAMBDA(typename Field::non_indices_domain_t::discrete_element_type elem) {
                int error = 0;
                auto const sample = sample_affine<
                        Capacity>([&](auto sampler) { return op.value(sampler, elem); }, error);
                if (error) {
                    system.fail(error);
                    return;
                }
                std::size_t const row = field.global_index(
                        elem,
                        ddc::DiscreteElement<typename Field::component_type>(0));
                if (row >= system.rows.extent(0))
                    system.fail(4);
                else
                    system.rows(row) = sample;
            });
}
} // namespace detail


/** Assemble scalar DEC stencils through topology-generated primal and dual rules.
 * \important This operator and documentation is fully AI-generated.
 * Each domain owns disjoint nodal rows and uses its partitioned execution space.
 * Dirichlet constraints include junction endpoints owned by connected domains.
 * Row stencils are packed into CSR on the device after all domains finish.
 */
template <class Direction, std::size_t ProbeCapacity = 0, class Domains, class Diffusivity>
ScalarLinearSystem assemble_multidomain_laplacian(
        Kokkos::DefaultExecutionSpace const& exec,
        Domains const& domains,
        std::size_t unknowns,
        Diffusivity diffusivity,
        double jump_scale = 1.0,
        bool normalize = true)
{
    auto const flux_domains
            = make_laplacian_flux_domains<Direction>(domains, diffusivity, jump_scale);
    sil::multidomains::
            DomainExecution<typename Domains::topology_type, Kokkos::DefaultExecutionSpace>
                    partitions(exec);
    constexpr std::size_t capacity = ProbeCapacity == 0
                                             ? (2 * Direction::size() + 1) * (Direction::size() + 1)
                                             : ProbeCapacity;
    detail::RowAssembly<capacity> const system(unknowns);
    partitions.for_each([&]<class Node>(Kokkos::DefaultExecutionSpace const& stream) {
        auto const field = domains.template field<typename Node::id>();
        auto const flux
                = flux_domains
                          .template field<typename Node::id, sil::multidomains::FieldRole::Flux>();
        auto const dual = flux_domains.template extrapolation_rules<
                typename Node::id,
                sil::multidomains::FieldRole::Flux>();
        sil::exterior::ScalarLaplacian<
                Direction,
                std::remove_cvref_t<decltype(flux)>,
                std::remove_cvref_t<decltype(dual)>> const op {flux, dual};
        detail::assemble_dec_rows<capacity>(stream, system, field, op);
        detail::prescribe_faces<capacity>(
                stream,
                system,
                field,
                domains.template extrapolation_rules<typename Node::id>(jump_scale),
                std::make_index_sequence<Direction::size()> {});
    });
    partitions.fence();
    return system.finalize(exec, normalize);
}
} // namespace similie::solvers
