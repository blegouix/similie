// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/chain.hpp>
#include <similie/exterior/local_chain.hpp>
#include <similie/exterior/reduction_and_reconstruction/reduction_and_reconstruction.hpp>

namespace sil::exterior {

namespace detail {

template <tensor::TensorIndex Index, class SimplexType>
KOKKOS_FUNCTION auto integration_element(SimplexType const& simplex)
{
    static_assert(Index::rank() == SimplexType::dimension());
    std::array<std::size_t, Index::rank()> directions {};
    std::size_t count = 0;
    auto const vector = ddc::detail::array(simplex.discrete_vector());
    for (std::size_t axis = 0; axis < vector.size(); ++axis) {
        if (vector[axis] != 0) {
            assert(count < directions.size());
            directions[count++] = axis;
        }
    }
    assert(count == directions.size());
    if constexpr (Index::rank() == 0) {
        return typename tensor::natural_domain_t<Index>::discrete_element_type(0);
    } else {
        return natural_elem_from_flat_ids<
                typename tensor::natural_domain_t<Index>::discrete_element_type>(directions);
    }
}

} // namespace detail

/** Evaluate an already reduced k-cochain on one oriented k-cell.
 * \important This operator and documentation is fully AI-generated.
 *
 * The tensor stores integrals on positively oriented cells, not physical
 * components. The cell's active logical directions select its coefficient;
 * its orientation supplies the sign. Logical axes must follow the same order
 * in the cell and the cochain index. No metric or cell measure is applied here.
 */
template <tensor::TensorIndex Index, class CochainTensor, class SimplexType>
KOKKOS_FUNCTION double integrate(CochainTensor const& cochain, SimplexType const& simplex)
{
    return (simplex.negative() ? -1.0 : 1.0)
           * cochain.get(cochain.access_element(
                   simplex.discrete_element(),
                   detail::integration_element<Index>(simplex)));
}

/** Pair a reduced k-cochain with an oriented k-chain: <c, C> = sum s_i c_i.
 * \important This operator and documentation is fully AI-generated.
 *
 * This is DEC integration. In particular, integrating df over C equals
 * integrating f over the boundary of C. An empty chain integrates to zero.
 * The execution space must be able to access both the chain and the tensor.
 */
template <tensor::TensorIndex Index, class ExecSpace, class CochainTensor, class ChainType>
double integrate(ExecSpace const& exec, CochainTensor const& cochain, ChainType const& chain)
{
    static_assert(
            Kokkos::SpaceAccessibility<ExecSpace, typename ChainType::memory_space>::accessible);
    static_assert(Kokkos::SpaceAccessibility<ExecSpace, typename CochainTensor::memory_space>::
                          accessible);
    double result = 0.0;
    Kokkos::parallel_reduce(
            "integrate_cochain",
            Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<std::size_t>>(exec, 0, chain.size()),
            KOKKOS_LAMBDA(std::size_t i, double& sum) {
                sum += integrate<Index>(cochain, chain[i]);
            },
            result);
    return result;
}

/** Reduce a physical differential form, then evaluate it on an oriented cell.
 * \important This operator and documentation is fully AI-generated.
 *
 * Reduction owns the geometric pullback and its sampling convention. This
 * overload adds no quadrature rule. Use the cochain overload when reduction
 * has already been performed, to avoid applying cell measures twice.
 */
template <misc::Specialization<ddc::TypeSeq> Indices, class Form, class Position, class SimplexType>
KOKKOS_FUNCTION double integrate(
        Form const& form,
        Position const& position,
        SimplexType const& simplex)
{
    using Operator = Reduction<Indices, Position, typename SimplexType::discrete_element_type>;
    using NaturalElement = typename tensor::tensor_accessor_for_domain_t<
            reduction_domain_t<Indices>>::natural_domain_t::discrete_element_type;
    if constexpr (ddc::type_seq_size_v<Indices> == 0) {
        return integrate<typename Operator::target_index_type>(form, simplex);
    } else {
        // Reduce only this cell's tangent component. Other forward directions may
        // lie outside the domain, for example on an upper boundary face.
        auto const target = detail::flat_natural_elem_ids(
                detail::integration_element<typename Operator::target_index_type>(simplex));
        [[maybe_unused]] tensor::TensorAccessor<typename Operator::source_index_type> source;
        double result = 0.0;
        ddc::device_for_each(source.natural_domain(), [&](auto source_element) {
            auto const coefficient = detail::merge_reduction_natural_elems<
                    NaturalElement>(detail::flat_natural_elem_ids(source_element), target);
            using FormElement = typename Form::accessor_t::natural_domain_t::discrete_element_type;
            FormElement const form_element = detail::natural_elem_from_flat_ids<FormElement>(
                    detail::flat_natural_elem_ids(source_element));
            result += Operator::value(position, simplex.discrete_element(), coefficient)
                      * form.get(form.access_element(simplex.discrete_element(), form_element));
        });
        return (simplex.negative() ? -1.0 : 1.0) * result;
    }
}

/** Reduce a physical form and pair it with a chain without storing a cochain.
 * \important This operator and documentation is fully AI-generated.
 */
template <
        misc::Specialization<ddc::TypeSeq> Indices,
        class ExecSpace,
        class Form,
        class Position,
        class ChainType>
double integrate(
        ExecSpace const& exec,
        Form const& form,
        Position const& position,
        ChainType const& chain)
{
    static_assert(
            Kokkos::SpaceAccessibility<ExecSpace, typename ChainType::memory_space>::accessible);
    static_assert(Kokkos::SpaceAccessibility<ExecSpace, typename Form::memory_space>::accessible);
    static_assert(
            Kokkos::SpaceAccessibility<ExecSpace, typename Position::memory_space>::accessible);
    double result = 0.0;
    Kokkos::parallel_reduce(
            "integrate_form",
            Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<std::size_t>>(exec, 0, chain.size()),
            KOKKOS_LAMBDA(std::size_t i, double& sum) {
                sum += integrate<Indices>(form, position, chain[i]);
            },
            result);
    return result;
}

} // namespace sil::exterior
