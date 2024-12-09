// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/filled_struct.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"


namespace sil {

namespace exterior {

namespace detail {

template <class T>
struct CoboundaryType;

template <std::size_t K, class... Tag, class ElementType, class LayoutStridedPolicy>
struct CoboundaryType<Cochain<Chain<Simplex<K, Tag...>>, ElementType, LayoutStridedPolicy>>
{
    using type = Cosimplex<Simplex<K + 1, Tag...>, ElementType>;
};

template <
        std::size_t K,
        class... Tag,
        class ElementType,
        class LayoutStridedPolicy1,
        class LayoutStridedPolicy2,
        class ExecSpace>
struct CoboundaryType<
        Cochain<Chain<Simplex<K, Tag...>, LayoutStridedPolicy1, ExecSpace>,
                ElementType,
                LayoutStridedPolicy2>>
{
    using type = Cosimplex<Simplex<K + 1, Tag...>, ElementType>;
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
using coboundary_t = typename detail::CoboundaryType<CochainType>::type;

namespace detail {

template <class TagToAddToCochain, class CochainTag>
struct CoboundaryIndex;

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorNatIndex CochainTag>
    requires(CochainTag::rank() == 0)
struct CoboundaryIndex<TagToAddToCochain, CochainTag>
{
    using type = TagToAddToCochain;
};

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorNatIndex CochainTag>
    requires(CochainTag::rank() == 1)
struct CoboundaryIndex<TagToAddToCochain, CochainTag>
{
    using type = tensor::TensorAntisymmetricIndex<TagToAddToCochain, CochainTag>;
};

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorNatIndex... Tag>
struct CoboundaryIndex<TagToAddToCochain, tensor::TensorAntisymmetricIndex<Tag...>>
{
    using type = tensor::TensorAntisymmetricIndex<TagToAddToCochain, Tag...>;
};

} // namespace detail

template <class TagToAddToCochain, class CochainTag>
using coboundary_index_t = typename detail::CoboundaryIndex<TagToAddToCochain, CochainTag>::type;

namespace detail {

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
struct CoboundaryTensorType;

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainIndex,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
struct CoboundaryTensorType<
        TagToAddToCochain,
        CochainIndex,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    static_assert(ddc::type_seq_contains_v<
                  ddc::detail::TypeSeq<CochainIndex>,
                  ddc::detail::TypeSeq<DDim...>>);
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    CochainIndex,
                    coboundary_index_t<TagToAddToCochain, CochainIndex>>,
            SupportType,
            MemorySpace>;
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
using coboundary_tensor_t =
        typename detail::CoboundaryTensorType<TagToAddToCochain, CochainTag, TensorType>::type;

namespace detail {

template <misc::Specialization<Chain> ChainType>
struct ComputeSimplex;

template <std::size_t K, class... Tag>
struct ComputeSimplex<Chain<Simplex<K, Tag...>>>
{
    KOKKOS_FUNCTION static Simplex<K + 1, Tag...> run(Chain<Simplex<K, Tag...>> const& chain)
    {
        ddc::DiscreteVector<Tag...> vect {
                0 * ddc::type_seq_rank_v<Tag, ddc::detail::TypeSeq<Tag...>>...};
        for (auto i = chain.begin(); i < chain.end(); ++i) {
            vect = ddc::DiscreteVector<Tag...> {
                    (static_cast<bool>(vect.template get<Tag>())
                     || static_cast<bool>((*i).discrete_vector().template get<Tag>()))...};
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

    /* Commented because would require an additional buffer
    assert(boundary(cochain.chain()) == boundary_t<typename CochainType::chain_type> {}
           && "only cochain over the boundary of a single simplex is supported");
     */

    return coboundary_t<CochainType>(
            detail::ComputeSimplex<typename CochainType::chain_type>::run(cochain.chain()),
            cochain.integrate());
}

namespace detail {

template <tensor::TensorNatIndex Index, class Dom>
struct NonSpectatorDimension;

template <tensor::TensorNatIndex Index, class... DDim>
struct NonSpectatorDimension<Index, ddc::DiscreteDomain<DDim...>>
{
    using type = ddc::cartesian_prod_t<std::conditional_t<
            ddc::type_seq_contains_v<
                    ddc::detail::TypeSeq<typename DDim::continuous_dimension_type>,
                    typename Index::type_seq_dimensions>,
            ddc::DiscreteDomain<DDim>,
            ddc::DiscreteDomain<>>...>;
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
KOKKOS_FUNCTION coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary(
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    ddc::for_each(
            ddc::remove_dims_of<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                    coboundary_tensor.domain()),
            [&](auto elem) {
                auto chain = tangent_basis<
                        CochainTag::rank() + 1,
                        typename detail::NonSpectatorDimension<
                                TagToAddToCochain,
                                typename TensorType::non_indices_domain_t>::type>();
                auto lower_chain = tangent_basis<
                        CochainTag::rank(),
                        typename detail::NonSpectatorDimension<
                                TagToAddToCochain,
                                typename TensorType::non_indices_domain_t>::type>();
                auto cochain = Cochain(chain, coboundary_tensor[elem]);
                for (auto i = cochain.begin(); i < cochain.end(); ++i) {
                    auto simplex = Simplex(
                            std::integral_constant<std::size_t, CochainTag::rank() + 1> {},
                            elem,
                            (*i).discrete_vector());
                    Kokkos::View<
                            simplex_for_domain_t<
                                    CochainTag::rank(),
                                    ddc::remove_dims_of_t<
                                            typename coboundary_tensor_t<
                                                    TagToAddToCochain,
                                                    CochainTag,
                                                    TensorType>::discrete_domain_type,
                                            coboundary_index_t<TagToAddToCochain, CochainTag>>>*,
                            Kokkos::LayoutRight,
                            Kokkos::HostSpace>
                            simplex_boundary_alloc("simplex_boundary", 2 * simplex.size());
                    sil::exterior::Chain simplex_boundary
                            = boundary(simplex_boundary_alloc, simplex);
                    Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::HostSpace>
                            values("coboundary_values", simplex_boundary.size());
                    for (auto j = simplex_boundary.begin(); j < simplex_boundary.end(); ++j) {
                        values(Kokkos::Experimental::distance(simplex_boundary.begin(), j))
                                = tensor.mem(
                                        misc::domain_contains(
                                                tensor.domain(),
                                                (*j).discrete_element())
                                                ? (*j).discrete_element()
                                                : elem, // TODO this is an assumption on boundary condition (free boundary), needs to be generalized
                                        ddc::DiscreteElement<CochainTag>(
                                                Kokkos::Experimental::distance(
                                                        lower_chain.begin(),
                                                        std::
                                                                find(lower_chain.begin(),
                                                                     lower_chain.end(),
                                                                     (*j).discrete_vector()))));
                    }
                    sil::exterior::Cochain cochain_boundary(simplex_boundary, values);
                    coboundary_tensor
                            .mem(elem,
                                 ddc::DiscreteElement<
                                         coboundary_index_t<TagToAddToCochain, CochainTag>>(
                                         Kokkos::Experimental::distance(cochain.begin(), i)))
                            = cochain_boundary.integrate();
                }
            });

    return coboundary_tensor;
}

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
KOKKOS_FUNCTION coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> deriv(
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    return coboundary<TagToAddToCochain, CochainTag, TensorType>(coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
