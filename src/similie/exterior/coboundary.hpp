// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/filled_struct.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/portable_stl.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/dummy_index.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "boundary.hpp"
#include "cochain.hpp"
#include "cosimplex.hpp"


namespace sil {

namespace exterior {

namespace detail {

template <class T>
struct CoboundaryType;

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

template <class SimplexType>
using simplex_boundary_type
        = std::array<boundary_t<SimplexType>, 2 * SimplexType::dimension()>;

template <class SimplexType>
KOKKOS_FUNCTION constexpr void generate_half_simplex_boundary(
        simplex_boundary_type<SimplexType>& simplex_boundary,
        std::size_t const offset,
        typename SimplexType::discrete_element_type elem,
        typename SimplexType::discrete_vector_type vect,
        bool const negative = false)
{
    auto array = ddc::detail::array(vect);
    auto id_dist = -1;
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        auto array_ = array;
        for (std::size_t j = id_dist + 1; j < array_.size(); ++j) {
            if (array_[j] != 0) {
                id_dist = static_cast<int>(j);
                array_[j] = 0;
                break;
            }
        }
        typename SimplexType::discrete_vector_type vect_;
        ddc::detail::array(vect_) = array_;
        simplex_boundary[offset + i]
                = boundary_t<SimplexType>(elem, vect_, (negative + i) % 2);
    }
}

template <class SimplexType>
KOKKOS_FUNCTION constexpr simplex_boundary_type<SimplexType> simplex_boundary(SimplexType simplex)
{
    simplex_boundary_type<SimplexType> simplex_boundary;

    generate_half_simplex_boundary<SimplexType>(
            simplex_boundary,
            0,
            simplex.discrete_element(),
            simplex.discrete_vector(),
            SimplexType::dimension() % 2);
    generate_half_simplex_boundary<SimplexType>(
            simplex_boundary,
            SimplexType::dimension(),
            simplex.discrete_element() + simplex.discrete_vector(),
            -simplex.discrete_vector());

    int const sign = (SimplexType::dimension() % 2 ? 1 : -1) * (simplex.negative() ? -1 : 1);
    if (sign == -1) {
        for (auto& boundary_simplex : simplex_boundary) {
            boundary_simplex = -boundary_simplex;
        }
    }

    return simplex_boundary;
}

template <misc::Specialization<LocalChain> ChainType, class DiscreteVectorType>
KOKKOS_FUNCTION std::size_t local_chain_index(
        ChainType const& chain,
        DiscreteVectorType const& vect)
{
    for (std::size_t i = 0; i < chain.size(); ++i) {
        if (chain[i].discrete_vector() == vect) {
            return i;
        }
    }
    assert(false && "simplex boundary vector must belong to the tangent basis");
    return 0;
}

template <
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        misc::Specialization<LocalChain> ChainType,
        class Elem,
        class SimplexType>
KOKKOS_FUNCTION typename TensorType::value_type coboundary_value(
        TensorType const& tensor,
        ChainType const& lower_chain,
        Elem const& elem,
        SimplexType const& simplex)
{
    typename TensorType::value_type out = 0.;
    auto const simplex_boundary = detail::simplex_boundary(simplex);

    for (auto const& boundary_simplex : simplex_boundary) {
        std::size_t const lower_chain_idx
                = local_chain_index(lower_chain, boundary_simplex.discrete_vector());
        out += (boundary_simplex.negative() ? -1. : 1.)
               * tensor.mem(
                       misc::domain_contains(tensor.domain(), boundary_simplex.discrete_element())
                               ? boundary_simplex.discrete_element()
                               : elem,
                       ddc::DiscreteElement<CochainTag>(lower_chain_idx));
    }

    return out;
}

template <misc::Specialization<Chain> ChainType>
struct ComputeSimplex;

template <std::size_t K, class... Tag, class LayoutStridedPolicy, class ExecSpace>
struct ComputeSimplex<Chain<Simplex<K, Tag...>, LayoutStridedPolicy, ExecSpace>>
{
    KOKKOS_FUNCTION static Simplex<K + 1, Tag...> run(
            Chain<Simplex<K, Tag...>, LayoutStridedPolicy, ExecSpace> const& chain)
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

struct CoboundaryDummyIndex
{
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary(
        ExecSpace const& exec_space,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    ddc::DiscreteDomain batch_dom
            = ddc::remove_dims_of<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                    coboundary_tensor.domain());

    // compute the tangent K+1-basis for each node of the mesh. This is a local K+1-chain.
    auto chain = tangent_basis<
            CochainTag::rank() + 1,
            typename detail::NonSpectatorDimension<
                    TagToAddToCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    // compute the tangent K-basis for each node of the mesh. This is a local K-chain.
    auto lower_chain = tangent_basis<
            CochainTag::rank(),
            typename detail::NonSpectatorDimension<
                    TagToAddToCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    // iterate over every node, we will work inside the tangent space associated to each of them
    SIMILIE_DEBUG_LOG("similie_compute_coboundary");
    ddc::parallel_for_each(
            "similie_compute_coboundary",
            exec_space,
            batch_dom,
            KOKKOS_LAMBDA(typename decltype(batch_dom)::discrete_element_type elem) {
                for (std::size_t i = 0; i < chain.size(); ++i) {
                    Simplex simplex = Simplex(
                            std::integral_constant<std::size_t, CochainTag::rank() + 1> {},
                            elem,
                            chain[i].discrete_vector());
                    coboundary_tensor
                            .mem(elem,
                                 ddc::DiscreteElement<
                                         coboundary_index_t<TagToAddToCochain, CochainTag>>(
                                         i))
                            = detail::coboundary_value<CochainTag>(
                                    tensor,
                                    lower_chain,
                                    elem,
                                    simplex);
                }
            });

    return coboundary_tensor;
}

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> deriv(
        ExecSpace const& exec_space,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    return coboundary<
            TagToAddToCochain,
            CochainTag,
            TensorType>(exec_space, coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
