// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "are_all_same.hpp"
#include "cochain.hpp"
#include "cosimplex.hpp"
#include "specialization.hpp"
#include "structured_cochain.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class T>
struct CoboundaryType;

template <std::size_t K, class... Tag, class ElementType, class Allocator>
struct CoboundaryType<Cochain<Chain<Simplex<K, Tag...>>, ElementType, Allocator>>
{
    using type = Cosimplex<Simplex<K + 1, Tag...>, ElementType>;
};

template <std::size_t K, class... Tag, class ElementType, class Allocator, class... Args>
struct CoboundaryType<
        StructuredCochain<Cochain<LocalChain<Simplex<K, Tag...>>, ElementType, Allocator>, Args...>>
{
    using type = StructuredCochain<
            Cochain<LocalChain<Simplex<K + 1, Tag...>>, ElementType, Allocator>,
            Args...>;
};

} // namespace detail

template <class T>
    requires(misc::Specialization<T, Cochain> || misc::Specialization<T, StructuredCochain>)
using coboundary_t = typename detail::CoboundaryType<T>::type;

namespace detail {

template <misc::Specialization<Chain> ChainType>
struct ComputeSimplex;

template <std::size_t K, class... Tag>
struct ComputeSimplex<Chain<Simplex<K, Tag...>>>
{
    static Simplex<K + 1, Tag...> run(Chain<Simplex<K, Tag...>> const& chain)
    {
        ddc::DiscreteVector<Tag...> vect {
                0 * ddc::type_seq_rank_v<Tag, ddc::detail::TypeSeq<Tag...>>...};
        for (auto i = chain.begin(); i < chain.end(); ++i) {
            vect = ddc::DiscreteVector<Tag...> {
                    (static_cast<bool>(vect.template get<Tag>())
                     || static_cast<bool>(i->discrete_vector().template get<Tag>()))...};
        }
        return Simplex(
                std::integral_constant<std::size_t, K + 1> {},
                chain[0].discrete_element(), // This is an assumption on the structure of the chain, which is satisfied if it has been produced using boundary()
                vect);
    }
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
KOKKOS_FUNCTION coboundary_t<CochainType> coboundary(
        CochainType
                cochain) // Warning: only cochain.chain() produced using boundary() are supported
{
    assert(cochain.size() == 2 * (cochain.dimension() + 1)
           && "only cochain over the boundary of a single simplex is supported");

    assert(boundary(cochain.chain()) == boundary_t<typename CochainType::chain_type> {}
           && "only cochain over the boundary of a single simplex is supported");

    return coboundary_t<CochainType>(
            detail::ComputeSimplex<typename CochainType::chain_type>::run(cochain.chain()),
            cochain.integrate());
}

template <misc::Specialization<StructuredCochain> StructuredCochainType>
KOKKOS_FUNCTION coboundary_t<StructuredCochainType> coboundary(
        coboundary_t<StructuredCochainType> structured_coboundary,
        StructuredCochainType structured_cochain)
{
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            structured_coboundary.domain(),
            [&](auto elem) {
                auto& cochain = structured_coboundary(elem);
                for (auto i = cochain.begin(); i < cochain.end(); ++i) {
                    sil::exterior::Chain simplex_boundary = boundary(
                            sil::exterior::Simplex(elem, *cochain.chain_it(i))); // TODO simplify
                    std::vector<double> values(simplex_boundary.size());
                    for (auto j = simplex_boundary.begin(); j < simplex_boundary.end(); ++j) {
                        values[std::distance(simplex_boundary.begin(), j)]
                                = structured_cochain(j->discrete_element(), j->discrete_vector());
                    }
                    sil::exterior::Cochain<decltype(simplex_boundary)>
                            cochain_boundary(simplex_boundary, values);
                    *i = cochain_boundary.integrate();
                }
            });

    return structured_coboundary;
}

KOKKOS_FUNCTION auto deriv(auto& cochain)
{
    return coboundary(cochain);
}

KOKKOS_FUNCTION auto deriv(auto& structured_coboundary, auto& structured_cochain)
{
    return coboundary(structured_coboundary, structured_cochain);
}

} // namespace exterior

} // namespace sil
