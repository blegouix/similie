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
#include <similie/misc/select_from_type_seq.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/dummy_index.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

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

template <class... Args>
struct Coboundary;

template <misc::Specialization<Cochain> CochainType>
struct Coboundary<CochainType>
{
    KOKKOS_FUNCTION static coboundary_t<CochainType>
    run(CochainType
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
};

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

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorIndex CochainTag>
struct PointwiseCoboundary
{
    template <
            class OutTensorType,
            class ValueAtFunc,
            class BatchDomain,
            class ChainType,
            class LowerChainType,
            class Elem>
    KOKKOS_FUNCTION static void run(
            OutTensorType coboundary_tensor,
            ValueAtFunc value_at,
            BatchDomain batch_domain,
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem)
    {
        constexpr std::size_t BOUNDARY_SIZE = 2 * (CochainTag::rank() + 1);
        using BoundarySimplex = typename LowerChainType::simplex_type;
        using MemorySpace = typename OutTensorType::memory_space;
        using BoundaryDomain = ddc::DiscreteDomain<CoboundaryDummyIndex>;
        using BoundaryChunk = ddc::
                ChunkSpan<BoundarySimplex, BoundaryDomain, Kokkos::layout_right, MemorySpace>;
        using BoundaryValuesChunk
                = ddc::ChunkSpan<double, BoundaryDomain, Kokkos::layout_right, MemorySpace>;

        std::array<BoundarySimplex, BOUNDARY_SIZE> simplex_boundary_storage {};
        std::array<double, BOUNDARY_SIZE> boundary_values_storage {};
        BoundaryDomain const boundary_domain(
                ddc::DiscreteElement<CoboundaryDummyIndex>(0),
                ddc::DiscreteVector<CoboundaryDummyIndex>(BOUNDARY_SIZE));
        BoundaryChunk simplex_boundary(simplex_boundary_storage.data(), boundary_domain);
        BoundaryValuesChunk boundary_values(boundary_values_storage.data(), boundary_domain);

        auto cochain = Cochain(chain, coboundary_tensor);
        for (auto i = cochain.begin(); i < cochain.end(); ++i) {
            using CoboundarySimplex = typename ChainType::simplex_type;
            using CoboundaryElement = typename CoboundarySimplex::discrete_element_type;
            CoboundarySimplex
                    simplex(std::integral_constant<std::size_t, CochainTag::rank() + 1> {},
                            CoboundaryElement(elem),
                            (*i).discrete_vector());
            auto boundary_chain = boundary(simplex_boundary.allocation_kokkos_view(), simplex);
            for (auto j = boundary_chain.begin(); j < boundary_chain.end(); ++j) {
                std::size_t const boundary_id
                        = Kokkos::Experimental::distance(boundary_chain.begin(), j);
                Elem sampled_elem = elem;
                if (misc::domain_contains(batch_domain, (*j).discrete_element())) {
                    using BoundaryElement = typename BoundarySimplex::discrete_element_type;
                    using SpectatorSeq = ddc::type_seq_remove_t<
                            ddc::to_type_seq_t<Elem>,
                            ddc::to_type_seq_t<BoundaryElement>>;
                    if constexpr (ddc::type_seq_size_v<SpectatorSeq> == 0) {
                        sampled_elem = Elem((*j).discrete_element());
                    } else {
                        sampled_elem
                                = Elem((*j).discrete_element(),
                                       misc::select_from_type_seq<SpectatorSeq>(elem));
                    }
                }
                boundary_values(ddc::DiscreteElement<CoboundaryDummyIndex>(boundary_id)) = value_at(
                        sampled_elem,
                        ddc::DiscreteElement<CochainTag>(Kokkos::Experimental::distance(
                                lower_chain.begin(),
                                misc::detail::
                                        find(lower_chain.begin(),
                                             lower_chain.end(),
                                             (*j).discrete_vector()))));
            }
            Cochain cochain_boundary(boundary_chain, boundary_values.allocation_kokkos_view());
            coboundary_tensor.mem(
                    ddc::DiscreteElement<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                            Kokkos::Experimental::distance(cochain.begin(), i)))
                    = cochain_boundary.integrate();
        }
    }
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
struct Coboundary<TagToAddToCochain, CochainTag, TensorType, ExecSpace>
{
    KOKKOS_FUNCTION static void run(
            auto coboundary_tensor,
            TensorType tensor,
            auto chain,
            auto lower_chain,
            auto elem)
    {
        auto value_at = [&](auto sampled_elem, auto cochain_elem) {
            return tensor.mem(sampled_elem, cochain_elem);
        };
        detail::PointwiseCoboundary<TagToAddToCochain, CochainTag>::
                run(coboundary_tensor,
                    value_at,
                    tensor.non_indices_domain(),
                    chain,
                    lower_chain,
                    elem);
    }

    static coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> run(
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
                    Coboundary::run(coboundary_tensor[elem], tensor, chain, lower_chain, elem);
                });

        return coboundary_tensor;
    }
};

template <misc::Specialization<Cochain> CochainType>
KOKKOS_FUNCTION coboundary_t<CochainType> coboundary(CochainType cochain)
{
    return Coboundary<CochainType>::run(cochain);
}

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
    return Coboundary<TagToAddToCochain, CochainTag, TensorType, ExecSpace>::
            run(exec_space, coboundary_tensor, tensor);
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
    return Coboundary<TagToAddToCochain, CochainTag, TensorType, ExecSpace>::
            run(exec_space, coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
