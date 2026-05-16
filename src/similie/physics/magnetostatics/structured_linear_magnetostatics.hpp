// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <ddc/ddc.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <similie/exterior/local_chain.hpp>
#include <similie/physics/hamilton_equations.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/physics/magnetostatics/magnetostatics_quantities.hpp>
#include <similie/physics/stationary_equations_operator.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

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

struct MagneticInductionValueFromPotential
{
    template <std::size_t I, class ChainType, class LowerChainType, class Elem>
    KOKKOS_INLINE_FUNCTION auto operator()(
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem) const
    {
        return MagneticVectorPotentialToMagneticInduction::template forward_value<I>(
                chain,
                lower_chain,
                elem);
    }
};

struct AssemblyRow
{
    static constexpr bool PERIODIC = false;
};

} // namespace detail

template <class MemorySpace>
class StructuredScalarPoissonStrongFormOperator2D
{
public:
    using memory_space = MemorySpace;
    using coord_view_type = Kokkos::View<double const*, memory_space>;

private:
    coord_view_type m_x_coords;
    coord_view_type m_y_coords;
    std::size_t m_nx;
    std::size_t m_ny;

public:
    StructuredScalarPoissonStrongFormOperator2D(coord_view_type x_coords, coord_view_type y_coords)
        : m_x_coords(x_coords)
        , m_y_coords(y_coords)
        , m_nx(x_coords.extent(0))
        , m_ny(y_coords.extent(0))
    {
    }

    StructuredScalarPoissonStrongFormOperator2D() : m_x_coords(), m_y_coords(), m_nx(0), m_ny(0) {}

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t size() const
    {
        return m_nx * m_ny;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t nx() const
    {
        return m_nx;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION std::size_t ny() const
    {
        return m_ny;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION coord_view_type x_coords() const
    {
        return m_x_coords;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION coord_view_type y_coords() const
    {
        return m_y_coords;
    }

    [[nodiscard]] KOKKOS_INLINE_FUNCTION bool is_boundary_node(std::size_t i, std::size_t j) const
    {
        return i == 0 || j == 0 || i + 1 == m_nx || j + 1 == m_ny;
    }

    template <class Functor>
    KOKKOS_INLINE_FUNCTION void for_each_nonzero_column(std::size_t row, Functor&& functor) const
    {
        std::size_t const j = row / m_nx;
        std::size_t const i = row % m_nx;
        if (is_boundary_node(i, j)) {
            functor(row);
            return;
        }
        functor(flat_index(i, j));
        functor(flat_index(i - 1, j));
        functor(flat_index(i + 1, j));
        functor(flat_index(i, j - 1));
        functor(flat_index(i, j + 1));
    }

    template <class InputView, class OutputView>
    KOKKOS_INLINE_FUNCTION void apply_at(OutputView output, InputView input, std::size_t row) const
    {
        std::size_t const j = row / m_nx;
        std::size_t const i = row % m_nx;

        bool const boundary = is_boundary_node(i, j);

        if (boundary) {
            output(row, 0) = input(row, 0);
            return;
        }

        double const dxm = m_x_coords(i) - m_x_coords(i - 1);
        double const dxp = m_x_coords(i + 1) - m_x_coords(i);
        double const dym = m_y_coords(j) - m_y_coords(j - 1);
        double const dyp = m_y_coords(j + 1) - m_y_coords(j);

        auto value = [&](std::size_t ii, std::size_t jj) { return input(flat_index(ii, jj), 0); };

        double const second_x
                = 2.0
                  * ((value(i + 1, j) - value(i, j)) / dxp - (value(i, j) - value(i - 1, j)) / dxm)
                  / (dxm + dxp);
        double const second_y
                = 2.0
                  * ((value(i, j + 1) - value(i, j)) / dyp - (value(i, j) - value(i, j - 1)) / dym)
                  / (dym + dyp);

        output(row, 0) = -(second_x + second_y);
    }

private:
    KOKKOS_INLINE_FUNCTION std::size_t flat_index(std::size_t i, std::size_t j) const
    {
        return i + m_nx * j;
    }
};
template <class Hamiltonian, class MemorySpace, class InputView, class OutputView>
KOKKOS_INLINE_FUNCTION void apply_stationary_equations_at(
        OutputView output,
        InputView input,
        std::size_t row,
        physics::HamiltonEquations<Hamiltonian> const& equations,
        StructuredScalarPoissonStrongFormOperator2D<MemorySpace> const& operator_model)
{
    // This matrix-free action is equations-driven: from A_z we reconstruct B,
    // evaluate H=dH/dB through the Hamiltonian, then discretize curl(H).
    std::size_t const nx = operator_model.nx();
    std::size_t const ny = operator_model.ny();
    std::size_t const j = row / nx;
    std::size_t const i = row % nx;

    bool const boundary = operator_model.is_boundary_node(i, j);
    if (boundary) {
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

    double const curl_h_z = 2.0 * (hy_e - hy_w) / (dxm + dxp) - 2.0 * (hx_n - hx_s) / (dym + dyp);
    output(row, 0) = curl_h_z;
}

template <class Hamiltonian, class MemorySpace>
KOKKOS_INLINE_FUNCTION double value_stationary_equations_at(
        std::size_t row,
        std::size_t column,
        physics::HamiltonEquations<Hamiltonian> const& equations,
        StructuredScalarPoissonStrongFormOperator2D<MemorySpace> const& operator_model)
{
    std::size_t const nx = operator_model.nx();
    std::size_t const ny = operator_model.ny();
    std::size_t const j = row / nx;
    std::size_t const i = row % nx;

    bool const boundary = operator_model.is_boundary_node(i, j);
    if (boundary) {
        return column == row ? 1.0 : 0.0;
    }

    auto flat_index = [nx](std::size_t ii, std::size_t jj) { return ii + nx * jj; };
    std::size_t const west = flat_index(i - 1, j);
    std::size_t const east = flat_index(i + 1, j);
    std::size_t const south = flat_index(i, j - 1);
    std::size_t const north = flat_index(i, j + 1);
    if (column != row && column != west && column != east && column != south && column != north) {
        return 0.0;
    }

    auto const x_coords = operator_model.x_coords();
    auto const y_coords = operator_model.y_coords();
    double const dxm = x_coords(i) - x_coords(i - 1);
    double const dxp = x_coords(i + 1) - x_coords(i);
    double const dym = y_coords(j) - y_coords(j - 1);
    double const dyp = y_coords(j + 1) - y_coords(j);

    double const coeff_hx
            = equations.template value<physics::PotentialTimeDerivative, 0>(0.0);
    double const coeff_hy
            = equations.template value<physics::PotentialTimeDerivative, 1>(0.0);

    if (column == west) {
        return -2.0 * coeff_hy / (dxm * (dxm + dxp));
    }
    if (column == east) {
        return -2.0 * coeff_hy / (dxp * (dxm + dxp));
    }
    if (column == south) {
        return -2.0 * coeff_hx / (dym * (dym + dyp));
    }
    if (column == north) {
        return -2.0 * coeff_hx / (dyp * (dym + dyp));
    }
    return 2.0 * coeff_hy / (dxm * dxp) + 2.0 * coeff_hx / (dym * dyp);
}

} // namespace similie::physics::magnetostatics

namespace similie::physics {

template <class PiComputerValue, class MemorySpace>
gko::matrix_data<double, gko::int32> assemble_matrix_data(
        StationaryEquationsOperator<
                HamiltonEquations<magnetostatics::LinearMagnetostaticsHamiltonian, PiComputerValue>,
                magnetostatics::StructuredScalarPoissonStrongFormOperator2D<MemorySpace>> const&
                operator_model)
{
    using OperatorType = StationaryEquationsOperator<
            HamiltonEquations<magnetostatics::LinearMagnetostaticsHamiltonian, PiComputerValue>,
            magnetostatics::StructuredScalarPoissonStrongFormOperator2D<MemorySpace>>;
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
                std::size_t const ny = structured_operator.ny();
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

                auto hx_s = operator_model.equations().template value<PotentialTimeDerivative, 0>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i, j - 1));
                auto hx_n = operator_model.equations().template value<PotentialTimeDerivative, 0>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i, j));
                auto hy_w = operator_model.equations().template value<PotentialTimeDerivative, 1>(
                        chain,
                        lower_chain,
                        ddc::DiscreteElement<magnetostatics::X, magnetostatics::Y>(i - 1, j));
                auto hy_e = operator_model.equations().template value<PotentialTimeDerivative, 1>(
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
    for (std::size_t row = 0; row < size; ++row) {
        for (int slot = 0; slot < counts_host(row); ++slot) {
            double const value = coefficients_host(row, slot);
            if (value != 0.0) {
                matrix_data.nonzeros.emplace_back(
                        static_cast<gko::int32>(row),
                        static_cast<gko::int32>(columns_host(row, slot)),
                        value);
            }
        }
    }
    matrix_data.sort_row_major();
    return matrix_data;
}

} // namespace similie::physics
