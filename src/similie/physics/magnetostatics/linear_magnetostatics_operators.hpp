// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>
#include <ginkgo/core/base/matrix_data.hpp>

#include <Kokkos_Core.hpp>

#include <similie/exterior/local_chain.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/stationary_equations_operator.hpp>
#include <similie/solvers/structured_scalar_poisson_strong_form_operator.hpp>

namespace similie::physics::magnetostatics {

using LinearMagnetostaticsOperator2D
        = solvers::StructuredScalarPoissonStrongFormOperator2D<X, Y, Kokkos::HostSpace>;

namespace detail {

struct LocalMagneticFieldTensor
{
    std::array<double, 3> values {0.0, 0.0, 0.0};

    template <class Dim>
    KOKKOS_INLINE_FUNCTION static constexpr std::size_t access_element()
    {
        if constexpr (std::is_same_v<Dim, X>) {
            return 0;
        } else if constexpr (std::is_same_v<Dim, Y>) {
            return 1;
        } else {
            return 2;
        }
    }

    KOKKOS_INLINE_FUNCTION double& operator()(std::size_t idx)
    {
        return values[idx];
    }

    KOKKOS_INLINE_FUNCTION double operator()(std::size_t idx) const
    {
        return values[idx];
    }
};

struct AssemblyRow
{
    static constexpr bool PERIODIC = false;
};

} // namespace detail

template <class Hamiltonian, class MemorySpace, class InputView, class OutputView>
KOKKOS_INLINE_FUNCTION void apply_stationary_equations_at(
        OutputView output,
        InputView input,
        std::size_t row,
        physics::HamiltonEquations<Hamiltonian> const& equations,
        solvers::StructuredScalarPoissonStrongFormOperator2D<X, Y, MemorySpace> const& operator_model)
{
    std::size_t const nx = operator_model.nx();
    std::size_t const ny = operator_model.ny();
    std::size_t const j = row / nx;
    std::size_t const i = row % nx;

    if (operator_model.is_boundary_node(i, j)) {
        output(row, 0) = input(row, 0);
        return;
    }

    auto flat_index = [nx](std::size_t ii, std::size_t jj) { return ii + nx * jj; };
    auto value = [&](std::size_t ii, std::size_t jj) { return input(flat_index(ii, jj), 0); };

    auto const x_coords = operator_model.x_coords();
    auto const y_coords = operator_model.y_coords();
    double const dxm = x_coords(i) - x_coords(i - 1);
    double const dxp = x_coords(i + 1) - x_coords(i);
    double const dym = y_coords(j) - y_coords(j - 1);
    double const dyp = y_coords(j + 1) - y_coords(j);

    auto magnetic_field_from_induction
            = [&](double bx, double by, double bz, std::size_t component) {
                  if (component == detail::LocalMagneticFieldTensor::template access_element<X>()) {
                      return equations.template dpotential_dt<0>(bx);
                  }
                  if (component == detail::LocalMagneticFieldTensor::template access_element<Y>()) {
                      return equations.template dpotential_dt<1>(by);
                  }
                  return equations.template dpotential_dt<2>(bz);
              };

    double const bx_s = (value(i, j) - value(i, j - 1)) / dym;
    double const bx_n = (value(i, j + 1) - value(i, j)) / dyp;
    double const by_w = -(value(i, j) - value(i - 1, j)) / dxm;
    double const by_e = -(value(i + 1, j) - value(i, j)) / dxp;

    double const hx_s = magnetic_field_from_induction(
            bx_s,
            0.0,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<X>());
    double const hx_n = magnetic_field_from_induction(
            bx_n,
            0.0,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<X>());
    double const hy_w = magnetic_field_from_induction(
            0.0,
            by_w,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<Y>());
    double const hy_e = magnetic_field_from_induction(
            0.0,
            by_e,
            0.0,
            detail::LocalMagneticFieldTensor::template access_element<Y>());

    output(row, 0) = 2.0 * (hy_e - hy_w) / (dxm + dxp) - 2.0 * (hx_n - hx_s) / (dym + dyp);
}

} // namespace similie::physics::magnetostatics

namespace similie::physics {

template <class PiComputerValue, class MemorySpace>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        StationaryEquationsOperator<
                HamiltonEquations<magnetostatics::LinearMagnetostaticsHamiltonian, PiComputerValue>,
                solvers::StructuredScalarPoissonStrongFormOperator2D<
                        magnetostatics::X,
                        magnetostatics::Y,
                        MemorySpace>> const& operator_model)
{
    using RowDomain = ddc::DiscreteDomain<magnetostatics::detail::AssemblyRow>;
    using SpatialDomain = ddc::DiscreteDomain<magnetostatics::X, magnetostatics::Y>;

    std::size_t const size = operator_model.size();
    Kokkos::DefaultExecutionSpace exec_space;
    RowDomain const row_domain(
            ddc::DiscreteElement<magnetostatics::detail::AssemblyRow>(0),
            ddc::DiscreteVector<magnetostatics::detail::AssemblyRow>(size));
    Kokkos::View<double*[5]> coefficients("similie_magnetostatics_matrix_coefficients", size);
    Kokkos::View<int*[5]> columns("similie_magnetostatics_matrix_columns", size);
    Kokkos::View<int*> counts("similie_magnetostatics_matrix_counts", size);

    auto chain = sil::exterior::tangent_basis<1, SpatialDomain>(exec_space);
    auto lower_chain = sil::exterior::tangent_basis<0, SpatialDomain>(exec_space);

    ddc::parallel_for_each(
            "similie_assemble_magnetostatics_matrix",
            exec_space,
            row_domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<magnetostatics::detail::AssemblyRow> row_elem) {
                std::size_t const row = row_elem.uid();
                auto const& structured_operator = operator_model.operator_model();
                std::size_t const nx = structured_operator.nx();
                std::size_t const j = row / nx;
                std::size_t const i = row % nx;

                auto flat_index = [nx](std::size_t ii, std::size_t jj) { return ii + nx * jj; };
                auto set_entry = [&](int slot, std::size_t column, double value) {
                    columns(row, slot) = static_cast<int>(column);
                    coefficients(row, slot) = value;
                };

                if (structured_operator.is_boundary_node(i, j)) {
                    counts(row) = 1;
                    set_entry(0, row, 1.0);
                    for (int slot = 1; slot < 5; ++slot) {
                        set_entry(slot, row, 0.0);
                    }
                    return;
                }

                auto const x_coords = structured_operator.x_coords();
                auto const y_coords = structured_operator.y_coords();
                double const dxm = x_coords(i) - x_coords(i - 1);
                double const dxp = x_coords(i + 1) - x_coords(i);
                double const dym = y_coords(j) - y_coords(j - 1);
                double const dyp = y_coords(j + 1) - y_coords(j);

                auto hx_s = operator_model.equations().template dpotential_dt_value<0>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i, j - 1));
                auto hx_n = operator_model.equations().template dpotential_dt_value<0>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i, j));
                auto hy_w = operator_model.equations().template dpotential_dt_value<1>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i - 1, j));
                auto hy_e = operator_model.equations().template dpotential_dt_value<1>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i, j));

                auto local_coeff = [&](auto const& stencil, std::size_t ii, std::size_t jj) {
                    auto const spatial_elem
                            = ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(ii, jj);
                    auto const scalar_elem = ddc::DiscreteElement<sil::tensor::ScalarIndex>(0);
                    if (!stencil.tensor.non_indices_domain().contains(spatial_elem)) {
                        return 0.0;
                    }
                    return stencil.tensor.get(spatial_elem, scalar_elem);
                };
                auto assembled_value = [&](std::size_t column) {
                    std::size_t const cj = column / nx;
                    std::size_t const ci = column % nx;
                    return 2.0 * (local_coeff(hy_e, ci, cj) - local_coeff(hy_w, ci, cj))
                                   / (dxm + dxp)
                           - 2.0 * (local_coeff(hx_n, ci, cj) - local_coeff(hx_s, ci, cj))
                                     / (dym + dyp);
                };

                counts(row) = 5;
                set_entry(0, flat_index(i, j), assembled_value(flat_index(i, j)));
                set_entry(1, flat_index(i - 1, j), assembled_value(flat_index(i - 1, j)));
                set_entry(2, flat_index(i + 1, j), assembled_value(flat_index(i + 1, j)));
                set_entry(3, flat_index(i, j - 1), assembled_value(flat_index(i, j - 1)));
                set_entry(4, flat_index(i, j + 1), assembled_value(flat_index(i, j + 1)));
            });
    exec_space.fence();

    auto coefficients_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficients);
    auto columns_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), columns);
    auto counts_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);

    gko::matrix_data<double, gko::int32> matrix_data(gko::dim<2>(size, size));
    matrix_data.nonzeros.reserve(size * 5);
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
    matrix_data.sort_row_major();
    return matrix_data;
}

} // namespace similie::physics
