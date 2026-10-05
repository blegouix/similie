// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/exterior/laplacian.hpp>
#include <similie/tensor/metric.hpp>
#include <similie/tensor/symmetric_tensor.hpp>

#include <Kokkos_Core.hpp>

namespace similie::onelab_interface::potential_flow_onelab {

struct X;
struct Y;
struct GridX
{
    using continuous_dimension_type = X;
};
struct GridY
{
    using continuous_dimension_type = Y;
};

using PotentialIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using GradientIndex = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;
using PositionIndex = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;
using MetricIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y>>>;

using GridDomain = ddc::DiscreteDomain<GridX, GridY>;
using PotentialDomain = ddc::DiscreteDomain<GridX, GridY, PotentialIndex>;
using PositionDomain = ddc::DiscreteDomain<GridX, GridY, PositionIndex>;
using MetricDomain = ddc::DiscreteDomain<GridX, GridY, MetricIndex>;

struct TensorLaplacianStencils
{
    std::vector<std::map<std::size_t, double>> rows;
    std::vector<std::map<std::size_t, double>> lower_x_flux_rows;
    std::vector<std::map<std::size_t, double>> upper_x_flux_rows;
    std::vector<std::map<std::size_t, double>> lower_y_flux_rows;
    std::vector<std::map<std::size_t, double>> upper_y_flux_rows;
};

/**
 * Extract the local stencil of the SimiLie scalar DEC Laplacian on one mapped
 * structured tensor domain.
 * \important This operator and documentation are fully AI-generated.
 *
 * The potential is a primal 0-cochain. SimiLie's staged Laplacian applies the
 * coboundary, the position- and metric-dependent discrete Hodge operators,
 * and the codifferential. Each color probes separated vertices, so a local
 * output coefficient identifies one column without implementing a second
 * Hodge star or a cell integration here. The caller supplies edge closure and
 * coupling equations when joining several tensor domains.
 */
inline TensorLaplacianStencils one_sided_tensor_laplacian_rows(
        std::vector<std::array<double, 2>> const& positions,
        std::size_t nodes_x,
        std::size_t nodes_y)
{
    constexpr std::size_t color_period = 7;
    constexpr int stencil_radius = 3;
    if (nodes_x < 2 || nodes_y < 2 || positions.size() != nodes_x * nodes_y)
        throw std::runtime_error("invalid tensor-domain position count");
    std::size_t const cells_x = nodes_x - 1;
    std::size_t const cells_y = nodes_y - 1;

    GridDomain const
            grid(ddc::DiscreteElement<GridX, GridY>(0, 0),
                 ddc::DiscreteVector<GridX, GridY>(nodes_x, nodes_y));
    GridDomain const cell_grid = grid.remove_last(ddc::DiscreteVector<GridX, GridY>(1, 1));
    [[maybe_unused]] sil::tensor::TensorAccessor<PotentialIndex> scalar_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    [[maybe_unused]] sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    PotentialDomain const potential_domain(grid, scalar_accessor.domain());
    PotentialDomain const laplacian_domain(cell_grid, scalar_accessor.domain());
    PositionDomain const position_domain(grid, position_accessor.domain());
    MetricDomain const metric_domain(grid, metric_accessor.domain());

    ddc::Chunk<double, PotentialDomain, ddc::DeviceAllocator<double>>
            potential_alloc(potential_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk<double, PotentialDomain, ddc::DeviceAllocator<double>>
            laplacian_alloc(laplacian_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk<double, PositionDomain, ddc::DeviceAllocator<double>>
            position_alloc(position_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk<double, MetricDomain, ddc::DeviceAllocator<double>>
            metric_alloc(metric_domain, ddc::DeviceAllocator<double>());
    ddc::Chunk<double, PotentialDomain, ddc::HostAllocator<double>>
            potential_host_alloc(potential_domain, ddc::HostAllocator<double>());
    ddc::Chunk<double, PotentialDomain, ddc::HostAllocator<double>>
            laplacian_host_alloc(laplacian_domain, ddc::HostAllocator<double>());
    ddc::Chunk<double, PositionDomain, ddc::HostAllocator<double>>
            position_host_alloc(position_domain, ddc::HostAllocator<double>());
    ddc::Chunk<double, MetricDomain, ddc::HostAllocator<double>>
            metric_host_alloc(metric_domain, ddc::HostAllocator<double>());

    sil::tensor::Tensor potential(potential_alloc);
    sil::tensor::Tensor laplacian(laplacian_alloc);
    sil::tensor::Tensor position(position_alloc);
    sil::tensor::Tensor metric(metric_alloc);
    sil::tensor::Tensor potential_host(potential_host_alloc);
    sil::tensor::Tensor laplacian_host(laplacian_host_alloc);
    sil::tensor::Tensor position_host(position_host_alloc);
    sil::tensor::Tensor metric_host(metric_host_alloc);

    for (std::size_t i = 0; i < nodes_x; ++i)
        for (std::size_t j = 0; j < nodes_y; ++j) {
            ddc::DiscreteElement<GridX, GridY> const elem(i, j);
            std::size_t const i0 = i == 0 ? i : i - 1;
            std::size_t const i1 = i + 1 == nodes_x ? i : i + 1;
            std::size_t const j0 = j == 0 ? j : j - 1;
            std::size_t const j1 = j + 1 == nodes_y ? j : j + 1;
            std::array<double, 2> const& left = positions[j * nodes_x + i0];
            std::array<double, 2> const& right = positions[j * nodes_x + i1];
            std::array<double, 2> const& lower = positions[j0 * nodes_x + i];
            std::array<double, 2> const& upper = positions[j1 * nodes_x + i];
            double const dx_x = (right[0] - left[0]) / static_cast<double>(i1 - i0);
            double const dx_y = (right[1] - left[1]) / static_cast<double>(i1 - i0);
            double const dy_x = (upper[0] - lower[0]) / static_cast<double>(j1 - j0);
            double const dy_y = (upper[1] - lower[1]) / static_cast<double>(j1 - j0);
            position_host(elem, position_accessor.access_element<X>()) = static_cast<double>(i);
            position_host(elem, position_accessor.access_element<Y>()) = static_cast<double>(j);
            metric_host(elem, metric_accessor.access_element<X, X>()) = dx_x * dx_x + dx_y * dx_y;
            metric_host(elem, metric_accessor.access_element<X, Y>()) = dx_x * dy_x + dx_y * dy_y;
            metric_host(elem, metric_accessor.access_element<Y, Y>()) = dy_x * dy_x + dy_y * dy_y;
        }
    ddc::parallel_deepcopy(position, position_host);
    ddc::parallel_deepcopy(metric, metric_host);

    auto staged_laplacian
            = sil::exterior::make_staged_laplacian<MetricIndex, GradientIndex, PotentialIndex>(
                    Kokkos::DefaultExecutionSpace(),
                    laplacian,
                    potential,
                    metric,
                    position);
    TensorLaplacianStencils stencils;
    stencils.rows.resize(nodes_x * nodes_y);
    stencils.lower_x_flux_rows.resize(nodes_y);
    stencils.upper_x_flux_rows.resize(nodes_y);
    stencils.lower_y_flux_rows.resize(nodes_x);
    stencils.upper_y_flux_rows.resize(nodes_x);
    for (std::size_t color_x = 0; color_x < color_period; ++color_x)
        for (std::size_t color_y = 0; color_y < color_period; ++color_y) {
            for (std::size_t i = 0; i < nodes_x; ++i)
                for (std::size_t j = 0; j < nodes_y; ++j)
                    potential_host.mem(ddc::DiscreteElement<GridX, GridY, PotentialIndex>(i, j, 0))
                            = (i % color_period == color_x && j % color_period == color_y) ? 1.0
                                                                                           : 0.0;
            ddc::parallel_deepcopy(potential, potential_host);
            staged_laplacian(laplacian, potential);
            Kokkos::fence();
            ddc::parallel_deepcopy(laplacian_host, laplacian);
            auto flux_host_alloc = ddc::create_mirror_view_and_copy(
                    Kokkos::DefaultHostExecutionSpace(),
                    staged_laplacian.derivative_dual_tensor_buffer());
            sil::tensor::Tensor flux_host(flux_host_alloc);
            for (std::size_t i = 0; i < cells_x; ++i)
                for (std::size_t j = 0; j < cells_y; ++j) {
                    double const value = laplacian_host.mem(
                            ddc::DiscreteElement<GridX, GridY, PotentialIndex>(i, j, 0));
                    if (std::abs(value) < 1.0e-14)
                        continue;
                    bool found = false;
                    for (int di = -stencil_radius; di <= stencil_radius; ++di)
                        for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
                            int const column_i = static_cast<int>(i) + di;
                            int const column_j = static_cast<int>(j) + dj;
                            if (column_i < 0 || column_i >= static_cast<int>(nodes_x)
                                || column_j < 0 || column_j >= static_cast<int>(nodes_y)
                                || static_cast<std::size_t>(column_i) % color_period != color_x
                                || static_cast<std::size_t>(column_j) % color_period != color_y)
                                continue;
                            if (found)
                                throw std::runtime_error("DEC stencil coloring is ambiguous");
                            stencils.rows[j * nodes_x + i][column_j * nodes_x + column_i] = value;
                            found = true;
                        }
                    if (!found)
                        throw std::runtime_error("DEC Laplacian stencil exceeds three cells");
                }
            for (std::size_t trace = 0; trace < 2; ++trace) {
                std::size_t const row_i = trace == 0 ? 0 : nodes_x - 2;
                std::vector<std::map<std::size_t, double>>& trace_rows
                        = trace == 0 ? stencils.lower_x_flux_rows : stencils.upper_x_flux_rows;
                for (std::size_t j = 0; j < nodes_y; ++j) {
                    ddc::DiscreteElement<GridX, GridY> const elem(row_i, j);
                    double value = 0.0;
                    ddc::host_for_each(flux_host.accessor().domain(), [&](auto component) {
                        auto const natural
                                = flux_host.accessor().canonical_natural_element(component);
                        if (ddc::detail::array(natural)[0] == 1)
                            value = flux_host.mem(typename decltype(flux_host)::
                                                          discrete_element_type(elem, component));
                    });
                    if (std::abs(value) < 1.0e-14)
                        continue;
                    bool found = false;
                    for (int di = -stencil_radius; di <= stencil_radius; ++di)
                        for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
                            int const column_i = static_cast<int>(row_i) + di;
                            int const column_j = static_cast<int>(j) + dj;
                            if (column_i < 0 || column_i >= static_cast<int>(nodes_x)
                                || column_j < 0 || column_j >= static_cast<int>(nodes_y)
                                || static_cast<std::size_t>(column_i) % color_period != color_x
                                || static_cast<std::size_t>(column_j) % color_period != color_y)
                                continue;
                            if (found)
                                throw std::runtime_error("DEC flux stencil coloring is ambiguous");
                            trace_rows[j][column_j * nodes_x + column_i] = value;
                            found = true;
                        }
                    if (!found)
                        throw std::runtime_error("DEC flux stencil exceeds three cells");
                }
            }
            for (std::size_t trace = 0; trace < 2; ++trace) {
                std::size_t const row_j = trace == 0 ? 0 : nodes_y - 2;
                std::vector<std::map<std::size_t, double>>& trace_rows
                        = trace == 0 ? stencils.lower_y_flux_rows : stencils.upper_y_flux_rows;
                for (std::size_t i = 0; i < nodes_x; ++i) {
                    ddc::DiscreteElement<GridX, GridY> const elem(i, row_j);
                    double value = 0.0;
                    ddc::host_for_each(flux_host.accessor().domain(), [&](auto component) {
                        auto const natural
                                = flux_host.accessor().canonical_natural_element(component);
                        if (ddc::detail::array(natural)[0] == 0)
                            value = flux_host.mem(typename decltype(flux_host)::
                                                          discrete_element_type(elem, component));
                    });
                    if (std::abs(value) < 1.0e-14)
                        continue;
                    bool found = false;
                    for (int di = -stencil_radius; di <= stencil_radius; ++di)
                        for (int dj = -stencil_radius; dj <= stencil_radius; ++dj) {
                            int const column_i = static_cast<int>(i) + di;
                            int const column_j = static_cast<int>(row_j) + dj;
                            if (column_i < 0 || column_i >= static_cast<int>(nodes_x)
                                || column_j < 0 || column_j >= static_cast<int>(nodes_y)
                                || static_cast<std::size_t>(column_i) % color_period != color_x
                                || static_cast<std::size_t>(column_j) % color_period != color_y)
                                continue;
                            if (found)
                                throw std::runtime_error(
                                        "DEC wall flux stencil coloring is ambiguous");
                            trace_rows[i][column_j * nodes_x + column_i] = value;
                            found = true;
                        }
                    if (!found)
                        throw std::runtime_error("DEC wall flux stencil exceeds three cells");
                }
            }
        }
    return stencils;
}

inline TensorLaplacianStencils tensor_laplacian_rows(
        std::vector<std::array<double, 2>> const& positions,
        std::size_t nodes_x,
        std::size_t nodes_y)
{
    TensorLaplacianStencils stencils = one_sided_tensor_laplacian_rows(positions, nodes_x, nodes_y);
    for (bool const reverse_x : {false, true})
        for (bool const reverse_y : {false, true}) {
            if (!reverse_x && !reverse_y)
                continue;
            std::vector<std::array<double, 2>> reversed_positions(positions.size());
            for (std::size_t j = 0; j < nodes_y; ++j)
                for (std::size_t i = 0; i < nodes_x; ++i) {
                    std::size_t const original_i = reverse_x ? nodes_x - 1 - i : i;
                    std::size_t const original_j = reverse_y ? nodes_y - 1 - j : j;
                    reversed_positions[j * nodes_x + i]
                            = positions[original_j * nodes_x + original_i];
                }
            TensorLaplacianStencils const reversed_stencils
                    = one_sided_tensor_laplacian_rows(reversed_positions, nodes_x, nodes_y);
            for (std::size_t j = 0; j < nodes_y; ++j)
                for (std::size_t i = 0; i < nodes_x; ++i) {
                    if ((reverse_x ? i != 0 : i == nodes_x - 1)
                        || (reverse_y ? j != 0 : j == nodes_y - 1))
                        continue;
                    std::size_t const mapped_i = reverse_x ? nodes_x - 1 - i : i;
                    std::size_t const mapped_j = reverse_y ? nodes_y - 1 - j : j;
                    std::map<std::size_t, double>& row
                            = stencils.rows[mapped_j * nodes_x + mapped_i];
                    for (auto const& [column, coefficient] :
                         reversed_stencils.rows[j * nodes_x + i]) {
                        std::size_t const column_i = column % nodes_x;
                        std::size_t const column_j = column / nodes_x;
                        std::size_t const mapped_column_i
                                = reverse_x ? nodes_x - 1 - column_i : column_i;
                        std::size_t const mapped_column_j
                                = reverse_y ? nodes_y - 1 - column_j : column_j;
                        row[mapped_column_j * nodes_x + mapped_column_i] = coefficient;
                    }
                }
        }
    return stencils;
}

} // namespace similie::onelab_interface::potential_flow_onelab
