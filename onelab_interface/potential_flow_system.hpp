// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace similie::onelab_interface::potential_flow_onelab {

struct PrescribedPotential
{
    double value;
};
struct NormalFlux
{
    double value = 0.0;
};
struct NaturalBoundary
{
};
using BoundaryCondition = std::variant<PrescribedPotential, NormalFlux, NaturalBoundary>;
enum class TraceSide { LowerX, UpperX };
struct TraceConnection
{
    std::size_t first_domain;
    TraceSide first_side;
    std::size_t second_domain;
    TraceSide second_side;
    double jump_coefficient = 0.0;
};

/** Data for one mapped tensor domain carrying a scalar 0-cochain. */
struct PotentialFlowPatch
{
    std::size_t nodes_x;
    std::size_t nodes_y;
    std::vector<std::array<double, 2>> positions;
    std::vector<std::map<std::size_t, double>> laplacian_rows;
    std::vector<std::map<std::size_t, double>> lower_x_flux_rows;
    std::vector<std::map<std::size_t, double>> upper_x_flux_rows;
    std::vector<std::map<std::size_t, double>> lower_y_flux_rows;
    std::vector<std::map<std::size_t, double>> upper_y_flux_rows;
    std::vector<std::size_t> ordering_key;
    std::vector<BoundaryCondition> lower_y;
    std::vector<BoundaryCondition> upper_y;
    bool conservative_laplacian_rows = false;
};

/**
 * A coupled scalar Laplacian with affine trace jumps.
 * \important This operator and documentation are fully AI-generated.
 *
 * The bulk rows come from the DEC Laplacian on each tensor domain. Interior
 * trace nodes are identified across a connection, with an affine potential
 * jump if requested. A second equation enforces equality of the normal flux.
 * Boundary conditions supply prescribed-potential, normal-flux, or
 * natural DEC equations on the remaining edges. Connected traces balance the
 * flux cochains produced by the first Hodge stage on each tensor domain.
 * The resulting square matrix acts on the identified primal 0-cochain; jump_rhs
 * is its response to a unit potential jump.
 */
struct PotentialFlowSystem
{
    std::vector<std::map<std::size_t, double>> rows;
    std::vector<double> base_rhs;
    std::vector<double> jump_rhs;
    std::vector<std::vector<std::size_t>> global_index;
    std::vector<std::vector<double>> jump_offset;
};

namespace detail {

class AffineTraceUnionFind
{
    std::vector<std::size_t> m_parent;
    std::vector<double> m_weight;

public:
    explicit AffineTraceUnionFind(std::size_t size) : m_parent(size), m_weight(size, 0.0)
    {
        for (std::size_t i = 0; i < size; ++i)
            m_parent[i] = i;
    }

    std::pair<std::size_t, double> find(std::size_t i)
    {
        if (m_parent[i] == i)
            return {i, 0.0};
        auto const [root, parent_weight] = find(m_parent[i]);
        m_weight[i] += parent_weight;
        m_parent[i] = root;
        return {root, m_weight[i]};
    }

    void connect(std::size_t first, std::size_t second, double jump)
    {
        auto const [first_root, first_weight] = find(first);
        auto const [second_root, second_weight] = find(second);
        if (first_root == second_root) {
            if (std::abs(first_weight - second_weight - jump) > 1.0e-12)
                throw std::runtime_error("inconsistent scalar trace jump");
            return;
        }
        m_parent[first_root] = second_root;
        m_weight[first_root] = jump - first_weight + second_weight;
    }
};

} // namespace detail

/**
 * Assemble the DEC bulk operator and the potential-flow boundary equations.
 * \important This operator and documentation are fully AI-generated.
 *
 * Domain boundaries are not inferred from mesh tags. The caller gives a condition
 * at every lower and upper y-boundary node and explicit connections between
 * x-traces. A connected trace shares one degree of freedom with its partner;
 * its affine offset carries the circulation cut. Explicit zero-flux
 * boundaries and connected traces use flux cochains from SimiLie's Hodge
 * stage. Natural boundaries use one-sided DEC Laplacian rows. A prescribed
 * nonzero flux is evaluated from primal edge differences, mapped edge vectors,
 * and the supplied free-scalar constitutive law.
 */
template <class X, class Y, class FluxLaw>
PotentialFlowSystem assemble_potential_flow_system(
        std::vector<PotentialFlowPatch> const& domains,
        std::vector<TraceConnection> const& connections,
        FluxLaw const& flux_law,
        double density)
{
    if (domains.empty() || !(density > 0.0))
        throw std::runtime_error("invalid coupled scalar domains or density");
    bool const keyed_order = !domains.front().ordering_key.empty();
    std::vector<std::size_t> starts(domains.size() + 1, 0);
    for (std::size_t side = 0; side < domains.size(); ++side) {
        PotentialFlowPatch const& domain = domains[side];
        if (domain.nodes_x < 2 || domain.nodes_y < 2
            || domain.positions.size() != domain.nodes_x * domain.nodes_y
            || domain.laplacian_rows.size() != domain.nodes_x * domain.nodes_y
            || domain.lower_x_flux_rows.size() != domain.nodes_y
            || domain.upper_x_flux_rows.size() != domain.nodes_y
            || domain.lower_y_flux_rows.size() != domain.nodes_x
            || domain.upper_y_flux_rows.size() != domain.nodes_x
            || (!domain.ordering_key.empty()
                && domain.ordering_key.size() != domain.nodes_x * domain.nodes_y)
            || domain.ordering_key.empty() == keyed_order || domain.lower_y.size() != domain.nodes_x
            || domain.upper_y.size() != domain.nodes_x)
            throw std::runtime_error("invalid scalar tensor domain");
        starts[side + 1] = starts[side] + domain.nodes_x * domain.nodes_y;
    }
    auto flat = [&](std::size_t side, std::size_t i, std::size_t j) {
        return starts[side] + j * domains[side].nodes_x + i;
    };
    auto trace_i = [&](std::size_t side, TraceSide trace) {
        return trace == TraceSide::LowerX ? std::size_t(0) : domains[side].nodes_x - 1;
    };
    detail::AffineTraceUnionFind trace_union(starts.back());
    for (TraceConnection const& connection : connections) {
        if (connection.first_domain >= domains.size() || connection.second_domain >= domains.size()
            || domains[connection.first_domain].nodes_y
                       != domains[connection.second_domain].nodes_y)
            throw std::runtime_error("incompatible connected scalar traces");
        std::size_t const first_i = trace_i(connection.first_domain, connection.first_side);
        std::size_t const second_i = trace_i(connection.second_domain, connection.second_side);
        for (std::size_t j = 0; j < domains[connection.first_domain].nodes_y; ++j)
            trace_union.connect(
                    flat(connection.first_domain, first_i, j),
                    flat(connection.second_domain, second_i, j),
                    connection.jump_coefficient);
    }

    PotentialFlowSystem system;
    system.global_index.resize(domains.size());
    system.jump_offset.resize(domains.size());
    std::vector<std::size_t> root_to_global(starts.back(), starts.back());
    std::vector<std::size_t> root_key(starts.back(), std::numeric_limits<std::size_t>::max());
    for (std::size_t side = 0; side < domains.size(); ++side) {
        std::size_t const local_size = domains[side].nodes_x * domains[side].nodes_y;
        for (std::size_t local = 0; local < local_size; ++local) {
            std::size_t const root = trace_union.find(starts[side] + local).first;
            std::size_t const key = domains[side].ordering_key.empty()
                                            ? starts[side] + local
                                            : domains[side].ordering_key[local];
            if (!keyed_order) {
                root_key[root] = std::min(root_key[root], key);
            } else {
                if (root_key[root] != std::numeric_limits<std::size_t>::max()
                    && root_key[root] != key)
                    throw std::runtime_error(
                            "connected scalar traces have different ordering keys");
                root_key[root] = key;
            }
        }
    }
    std::vector<std::pair<std::size_t, std::size_t>> ordered_roots;
    for (std::size_t root = 0; root < starts.back(); ++root)
        if (root_key[root] != std::numeric_limits<std::size_t>::max())
            ordered_roots.emplace_back(root_key[root], root);
    std::sort(ordered_roots.begin(), ordered_roots.end());
    for (std::size_t index = 0; index < ordered_roots.size(); ++index) {
        if (index > 0 && ordered_roots[index - 1].first == ordered_roots[index].first)
            throw std::runtime_error("duplicate scalar degree-of-freedom ordering key");
        root_to_global[ordered_roots[index].second] = index;
    }
    std::size_t const global_size = ordered_roots.size();
    for (std::size_t side = 0; side < domains.size(); ++side) {
        std::size_t const local_size = domains[side].nodes_x * domains[side].nodes_y;
        system.global_index[side].resize(local_size);
        system.jump_offset[side].resize(local_size);
        for (std::size_t local = 0; local < local_size; ++local) {
            auto const [root, offset] = trace_union.find(starts[side] + local);
            system.global_index[side][local] = root_to_global[root];
            system.jump_offset[side][local] = offset;
        }
    }
    system.rows.resize(global_size);
    system.base_rhs.resize(global_size, 0.0);
    system.jump_rhs.resize(global_size, 0.0);
    std::vector<bool> assigned(global_size, false);
    auto global = [&](std::size_t side, std::size_t i, std::size_t j) {
        return system.global_index[side][j * domains[side].nodes_x + i];
    };
    auto claim = [&](std::size_t row) {
        if (assigned[row])
            throw std::runtime_error("duplicate coupled scalar equation");
        assigned[row] = true;
    };
    auto add = [&](std::size_t row,
                   std::size_t side,
                   std::size_t i,
                   std::size_t j,
                   double coefficient) {
        std::size_t const local = j * domains[side].nodes_x + i;
        system.rows[row][system.global_index[side][local]] += coefficient;
        system.jump_rhs[row] -= coefficient * system.jump_offset[side][local];
    };
    auto position = [&](std::size_t side, std::size_t i, std::size_t j) {
        return domains[side].positions[j * domains[side].nodes_x + i];
    };
    auto add_dec_row = [&](std::size_t row, std::size_t side, std::size_t i, std::size_t j) {
        PotentialFlowPatch const& domain = domains[side];
        std::map<std::size_t, double> const& local_row
                = domain.laplacian_rows[j * domain.nodes_x + i];
        if (local_row.empty())
            throw std::runtime_error("missing DEC scalar boundary row");
        for (auto const& [column, coefficient] : local_row)
            add(row, side, column % domain.nodes_x, column / domain.nodes_x, coefficient);
    };
    auto normal_flux = [&](std::size_t row,
                           std::size_t side,
                           std::size_t ti0,
                           std::size_t tj0,
                           std::size_t ti1,
                           std::size_t tj1,
                           std::size_t ri0,
                           std::size_t rj0,
                           std::size_t ri1,
                           std::size_t rj1,
                           double sign) {
        std::array<double, 2> const tm = position(side, ti0, tj0);
        std::array<double, 2> const tp = position(side, ti1, tj1);
        std::array<double, 2> const rm = position(side, ri0, rj0);
        std::array<double, 2> const rp = position(side, ri1, rj1);
        double const tx = tp[0] - tm[0], ty = tp[1] - tm[1];
        double const rx = rp[0] - rm[0], ry = rp[1] - rm[1];
        double const determinant = tx * ry - ty * rx;
        double const tangent_length = std::hypot(tx, ty);
        if (std::abs(determinant) < 1.0e-14 || tangent_length < 1.0e-14)
            throw std::runtime_error("degenerate scalar boundary frame");
        double const nx = ty / tangent_length, ny = -tx / tangent_length;
        double const tangential_weight
                = sign * density
                  * (nx * flux_law.template dpotential_dt<X>(ry / determinant)
                     + ny * flux_law.template dpotential_dt<Y>(-rx / determinant));
        double const transverse_weight
                = sign * density
                  * (nx * flux_law.template dpotential_dt<X>(-ty / determinant)
                     + ny * flux_law.template dpotential_dt<Y>(tx / determinant));
        add(row, side, ti1, tj1, tangential_weight);
        add(row, side, ti0, tj0, -tangential_weight);
        add(row, side, ri1, rj1, transverse_weight);
        add(row, side, ri0, rj0, -transverse_weight);
    };

    auto wall_flux = [&](std::size_t row, std::size_t side, std::size_t i, std::size_t j) {
        PotentialFlowPatch const& domain = domains[side];
        std::size_t const i0 = i == 0 ? 0 : i - 1;
        std::size_t const i1 = i + 1 == domain.nodes_x ? i : i + 1;
        std::size_t const inner_j = j == 0 ? 1 : j - 1;
        std::array<double, 2> const p0 = position(side, i0, j);
        std::array<double, 2> const p1 = position(side, i1, j);
        std::array<double, 2> const wall = position(side, i, j);
        std::array<double, 2> const inner = position(side, i, inner_j);
        double const determinant
                = (p1[0] - p0[0]) * (inner[1] - wall[1]) - (p1[1] - p0[1]) * (inner[0] - wall[0]);
        normal_flux(row, side, i0, j, i1, j, i, j, i, inner_j, determinant > 0.0 ? 1.0 : -1.0);
    };

    for (std::size_t side = 0; side < domains.size(); ++side) {
        PotentialFlowPatch const& domain = domains[side];
        for (std::size_t i = 1; i + 1 < domain.nodes_x; ++i)
            for (std::size_t j = 1; j + 1 < domain.nodes_y; ++j) {
                std::size_t const row = global(side, i, j);
                claim(row);
                for (auto const& [column, coefficient] :
                     domain.laplacian_rows[j * domain.nodes_x + i])
                    add(row, side, column % domain.nodes_x, column / domain.nodes_x, coefficient);
            }
        for (std::size_t i = 1; i + 1 < domain.nodes_x; ++i)
            for (std::size_t boundary_id = 0; boundary_id < 2; ++boundary_id) {
                std::size_t const j = boundary_id == 0 ? 0 : domain.nodes_y - 1;
                BoundaryCondition const& rule
                        = boundary_id == 0 ? domain.lower_y[i] : domain.upper_y[i];
                std::size_t const row = global(side, i, j);
                claim(row);
                if (auto const* prescribed = std::get_if<PrescribedPotential>(&rule)) {
                    add(row, side, i, j, 1.0);
                    system.base_rhs[row] = prescribed->value;
                } else if (std::holds_alternative<NaturalBoundary>(rule)) {
                    add_dec_row(row, side, i, j);
                } else {
                    auto const& flux = std::get<NormalFlux>(rule);
                    wall_flux(row, side, i, j);
                    system.base_rhs[row] = flux.value;
                }
            }
    }
    for (TraceConnection const& connection : connections) {
        std::size_t const first = connection.first_domain;
        std::size_t const second = connection.second_domain;
        if (connection.first_side != TraceSide::UpperX
            || connection.second_side != TraceSide::LowerX)
            throw std::runtime_error("connected scalar traces must meet upper X to lower X");
        if (domains[first].conservative_laplacian_rows
            != domains[second].conservative_laplacian_rows)
            throw std::runtime_error("connected scalar domains use different row conventions");
        std::size_t const first_i = trace_i(first, connection.first_side);
        std::size_t const second_i = trace_i(second, connection.second_side);
        for (std::size_t j = 0; j < domains[first].nodes_y; ++j)
            if (std::
                        hypot(position(first, first_i, j)[0] - position(second, second_i, j)[0],
                              position(first, first_i, j)[1] - position(second, second_i, j)[1])
                > 1.0e-10)
                throw std::runtime_error("connected scalar traces have different positions");
        for (std::size_t j = 0; j < domains[first].nodes_y; ++j) {
            std::size_t const row = global(first, first_i, j);
            claim(row);
            if (j == 0 || j + 1 == domains[first].nodes_y) {
                BoundaryCondition const& first_rule = j == 0 ? domains[first].lower_y[first_i]
                                                             : domains[first].upper_y[first_i];
                BoundaryCondition const& second_rule = j == 0 ? domains[second].lower_y[second_i]
                                                              : domains[second].upper_y[second_i];
                auto const* first_prescribed = std::get_if<PrescribedPotential>(&first_rule);
                auto const* second_prescribed = std::get_if<PrescribedPotential>(&second_rule);
                if (first_prescribed != nullptr || second_prescribed != nullptr) {
                    if (first_prescribed != nullptr && second_prescribed != nullptr
                        && std::abs(first_prescribed->value - second_prescribed->value) > 1.0e-12)
                        throw std::runtime_error(
                                "incompatible prescribed scalar values at connected trace");
                    if (first_prescribed != nullptr) {
                        add(row, first, first_i, j, 1.0);
                        system.base_rhs[row] = first_prescribed->value;
                    } else {
                        add(row, second, second_i, j, 1.0);
                        system.base_rhs[row] = second_prescribed->value;
                    }
                    continue;
                }
                // At a wall endpoint the shared unknown still needs a wall
                // equation. Matching the interface flux here would drop both
                // physical boundary rules.
                for (std::size_t const side : {first, second}) {
                    PotentialFlowPatch const& domain = domains[side];
                    std::size_t const i = side == first ? first_i : second_i;
                    BoundaryCondition const& rule = side == first ? first_rule : second_rule;
                    if (std::holds_alternative<NaturalBoundary>(rule)) {
                        add_dec_row(row, side, i, j);
                    } else {
                        NormalFlux const& flux = std::get<NormalFlux>(rule);
                        wall_flux(row, side, i, j);
                        system.base_rhs[row] += flux.value;
                    }
                }
                continue;
            }
            if (domains[first].conservative_laplacian_rows) {
                add_dec_row(row, first, first_i, j);
                add_dec_row(row, second, second_i, j);
                continue;
            }
            normal_flux(
                    row,
                    first,
                    first_i,
                    j - 1,
                    first_i,
                    j + 1,
                    first_i - 1,
                    j,
                    first_i,
                    j,
                    1.0);
            normal_flux(
                    row,
                    second,
                    second_i,
                    j - 1,
                    second_i,
                    j + 1,
                    second_i,
                    j,
                    second_i + 1,
                    j,
                    -1.0);
        }
    }
    for (std::size_t row = 0; row < global_size; ++row) {
        if (!assigned[row] || system.rows[row].empty())
            throw std::runtime_error("missing coupled scalar equation");
        double scale = 0.0;
        for (auto const& [column, coefficient] : system.rows[row])
            scale = std::max(scale, std::abs(coefficient));
        if (scale < 1.0e-16)
            throw std::runtime_error("zero coupled scalar equation");
        for (auto& [column, coefficient] : system.rows[row])
            coefficient /= scale;
        system.base_rhs[row] /= scale;
        system.jump_rhs[row] /= scale;
    }
    return system;
}

} // namespace similie::onelab_interface::potential_flow_onelab
