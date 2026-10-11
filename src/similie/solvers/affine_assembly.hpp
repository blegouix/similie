// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <limits>

#include "sparse_linear_system.hpp"

namespace similie::solvers {
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
        return {SparseLinearOperator(offsets, columns, values), rhs};
    }
};
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
} // namespace similie::solvers
