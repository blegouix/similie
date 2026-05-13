// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/clamp_to_domain.hpp>
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

template <class... Args>
struct TransposedCoboundary;

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

} // namespace detail

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorIndex CochainTag>
struct Coboundary<TagToAddToCochain, CochainTag>
{
    template <
            class CoboundaryTensorType,
            class Evaluator,
            class ChainType,
            class LowerChainType,
            class Elem>
    KOKKOS_FUNCTION static void run(
            CoboundaryTensorType coboundary_tensor,
            Evaluator evaluator,
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem)
    {
        constexpr std::size_t BOUNDARY_SIZE = 2 * (CochainTag::rank() + 1);
        std::array<typename LowerChainType::simplex_type, BOUNDARY_SIZE> simplex_boundary_alloc {};
        std::array<double, BOUNDARY_SIZE> boundary_values_alloc {};
        ddc::DiscreteDomain<detail::CoboundaryDummyIndex> const boundary_domain(
                ddc::DiscreteElement<detail::CoboundaryDummyIndex>(0),
                ddc::DiscreteVector<detail::CoboundaryDummyIndex>(BOUNDARY_SIZE));
        ddc::ChunkSpan<
                typename LowerChainType::simplex_type,
                ddc::DiscreteDomain<detail::CoboundaryDummyIndex>,
                Kokkos::layout_right,
                typename CoboundaryTensorType::memory_space>
                simplex_boundary(simplex_boundary_alloc.data(), boundary_domain);
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<detail::CoboundaryDummyIndex>,
                Kokkos::layout_right,
                typename CoboundaryTensorType::memory_space>
                boundary_values(boundary_values_alloc.data(), boundary_domain);

        auto cochain = Cochain(chain, coboundary_tensor);
        for (auto i = cochain.begin(); i < cochain.end(); ++i) {
            typename ChainType::simplex_type
                    simplex(std::integral_constant<std::size_t, CochainTag::rank() + 1> {},
                            typename ChainType::simplex_type::discrete_element_type(elem),
                            (*i).discrete_vector());
            auto boundary_chain = boundary(simplex_boundary.allocation_kokkos_view(), simplex);
            for (auto j = boundary_chain.begin(); j < boundary_chain.end(); ++j) {
                std::size_t const boundary_id
                        = Kokkos::Experimental::distance(boundary_chain.begin(), j);
                if constexpr (
                        ddc::type_seq_size_v<ddc::type_seq_remove_t<
                                ddc::to_type_seq_t<Elem>,
                                ddc::to_type_seq_t<typename LowerChainType::simplex_type::
                                                           discrete_element_type>>>
                        == 0) {
                    boundary_values(ddc::DiscreteElement<detail::CoboundaryDummyIndex>(boundary_id))
                            = evaluator(
                                    Elem((*j).discrete_element()),
                                    ddc::DiscreteElement<CochainTag>(Kokkos::Experimental::distance(
                                            lower_chain.begin(),
                                            misc::detail::
                                                    find(lower_chain.begin(),
                                                         lower_chain.end(),
                                                         (*j).discrete_vector()))));
                } else {
                    boundary_values(ddc::DiscreteElement<detail::CoboundaryDummyIndex>(boundary_id))
                            = evaluator(
                                    Elem((*j).discrete_element(),
                                         misc::select_from_type_seq<ddc::type_seq_remove_t<
                                                 ddc::to_type_seq_t<Elem>,
                                                 ddc::to_type_seq_t<
                                                         typename LowerChainType::simplex_type::
                                                                 discrete_element_type>>>(elem)),
                                    ddc::DiscreteElement<CochainTag>(Kokkos::Experimental::distance(
                                            lower_chain.begin(),
                                            misc::detail::
                                                    find(lower_chain.begin(),
                                                         lower_chain.end(),
                                                         (*j).discrete_vector()))));
                }
            }
            Cochain cochain_boundary(boundary_chain, boundary_values.allocation_kokkos_view());
            coboundary_tensor.mem(
                    ddc::DiscreteElement<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                            Kokkos::Experimental::distance(cochain.begin(), i)))
                    = cochain_boundary.integrate();
        }
    }
};

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorIndex CochainTag>
struct TransposedCoboundary<TagToAddToCochain, CochainTag>
{
    template <
            class CoboundaryTensorType,
            class Evaluator,
            class ChainType,
            class LowerChainType,
            class Elem>
    KOKKOS_FUNCTION static void run(
            CoboundaryTensorType coboundary_tensor,
            Evaluator evaluator,
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem)
    {
        constexpr std::size_t BOUNDARY_SIZE = 2 * (CochainTag::rank() + 1);
        std::array<typename LowerChainType::simplex_type, BOUNDARY_SIZE> simplex_boundary_alloc {};
        std::array<double, BOUNDARY_SIZE> boundary_values_alloc {};
        ddc::DiscreteDomain<detail::CoboundaryDummyIndex> const boundary_domain(
                ddc::DiscreteElement<detail::CoboundaryDummyIndex>(0),
                ddc::DiscreteVector<detail::CoboundaryDummyIndex>(BOUNDARY_SIZE));
        ddc::ChunkSpan<
                typename LowerChainType::simplex_type,
                ddc::DiscreteDomain<detail::CoboundaryDummyIndex>,
                Kokkos::layout_right,
                typename CoboundaryTensorType::memory_space>
                simplex_boundary(simplex_boundary_alloc.data(), boundary_domain);
        ddc::ChunkSpan<
                double,
                ddc::DiscreteDomain<detail::CoboundaryDummyIndex>,
                Kokkos::layout_right,
                typename CoboundaryTensorType::memory_space>
                boundary_values(boundary_values_alloc.data(), boundary_domain);

        auto cochain = Cochain(chain, coboundary_tensor);
        for (auto i = cochain.begin(); i < cochain.end(); ++i) {
            typename ChainType::simplex_type
                    simplex(std::integral_constant<std::size_t, CochainTag::rank() + 1> {},
                            typename ChainType::simplex_type::discrete_element_type(elem),
                            (*i).discrete_vector());
            auto boundary_chain = boundary(simplex_boundary.allocation_kokkos_view(), simplex);
            for (auto j = boundary_chain.begin(); j < boundary_chain.end(); ++j) {
                std::size_t const boundary_id
                        = Kokkos::Experimental::distance(boundary_chain.begin(), j);
                auto sampled_face_vector = (*j).discrete_vector();
                auto sampled_face_elem = (*j).discrete_element();
                if (boundary_id >= CochainTag::rank() + 1) {
                    auto simplex_vector = (*i).discrete_vector();
                    for (std::size_t dim_id = 0;
                         dim_id < ddc::type_seq_size_v<ddc::to_type_seq_t<
                                 typename ChainType::simplex_type::discrete_element_type>>;
                         ++dim_id) {
                        if (ddc::detail::array(simplex_vector)[dim_id] != 0
                            && ddc::detail::array(sampled_face_vector)[dim_id] == 0) {
                            ddc::detail::array(sampled_face_elem)[dim_id] -= 2;
                            break;
                        }
                    }
                }
                if constexpr (
                        ddc::type_seq_size_v<ddc::type_seq_remove_t<
                                ddc::to_type_seq_t<Elem>,
                                ddc::to_type_seq_t<typename LowerChainType::simplex_type::
                                                           discrete_element_type>>>
                        == 0) {
                    boundary_values(ddc::DiscreteElement<detail::CoboundaryDummyIndex>(boundary_id))
                            = evaluator(
                                    Elem(sampled_face_elem),
                                    ddc::DiscreteElement<CochainTag>(Kokkos::Experimental::distance(
                                            lower_chain.begin(),
                                            misc::detail::
                                                    find(lower_chain.begin(),
                                                         lower_chain.end(),
                                                         sampled_face_vector))));
                } else {
                    boundary_values(ddc::DiscreteElement<detail::CoboundaryDummyIndex>(boundary_id))
                            = evaluator(
                                    Elem(sampled_face_elem,
                                         misc::select_from_type_seq<ddc::type_seq_remove_t<
                                                 ddc::to_type_seq_t<Elem>,
                                                 ddc::to_type_seq_t<
                                                         typename LowerChainType::simplex_type::
                                                                 discrete_element_type>>>(elem)),
                                    ddc::DiscreteElement<CochainTag>(Kokkos::Experimental::distance(
                                            lower_chain.begin(),
                                            misc::detail::
                                                    find(lower_chain.begin(),
                                                         lower_chain.end(),
                                                         sampled_face_vector))));
                }
            }
            Cochain cochain_boundary(boundary_chain, boundary_values.allocation_kokkos_view());
            coboundary_tensor.mem(
                    ddc::DiscreteElement<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                            Kokkos::Experimental::distance(cochain.begin(), i)))
                    = -cochain_boundary.integrate();
        }
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
    ddc::DiscreteDomain batch_dom
            = ddc::remove_dims_of<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                    coboundary_tensor.domain());

    auto chain = tangent_basis<
            CochainTag::rank() + 1,
            typename detail::NonSpectatorDimension<
                    TagToAddToCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto lower_chain = tangent_basis<
            CochainTag::rank(),
            typename detail::NonSpectatorDimension<
                    TagToAddToCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    SIMILIE_DEBUG_LOG("similie_compute_coboundary");
    ddc::parallel_for_each(
            "similie_compute_coboundary",
            exec_space,
            batch_dom,
            KOKKOS_LAMBDA(typename decltype(batch_dom)::discrete_element_type elem) {
                Coboundary<TagToAddToCochain, CochainTag>::run(
                        coboundary_tensor[elem],
                        // TODO this is an assumption on boundary condition (free boundary), needs to be generalized.
                        [&](auto sampled_elem, auto cochain_elem) {
                            auto const clamped_elem = misc::
                                    clamp_to_domain(tensor.non_indices_domain(), sampled_elem);
                            return tensor.mem(clamped_elem, cochain_elem);
                        },
                        chain,
                        lower_chain,
                        elem);
            });

    return coboundary_tensor;
}

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> transposed_coboundary(
        ExecSpace const& exec_space,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    ddc::DiscreteDomain batch_dom
            = ddc::remove_dims_of<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                    coboundary_tensor.domain());

    auto chain = tangent_basis<
            CochainTag::rank() + 1,
            typename detail::NonSpectatorDimension<
                    TagToAddToCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);
    auto lower_chain = tangent_basis<
            CochainTag::rank(),
            typename detail::NonSpectatorDimension<
                    TagToAddToCochain,
                    typename TensorType::non_indices_domain_t>::type>(exec_space);

    SIMILIE_DEBUG_LOG("similie_compute_transposed_coboundary");
    ddc::parallel_for_each(
            "similie_compute_transposed_coboundary",
            exec_space,
            batch_dom,
            KOKKOS_LAMBDA(typename decltype(batch_dom)::discrete_element_type elem) {
                TransposedCoboundary<TagToAddToCochain, CochainTag>::run(
                        coboundary_tensor[elem],
                        [&](auto sampled_elem, auto cochain_elem) {
                            if (!misc::domain_contains(tensor.non_indices_domain(), sampled_elem)) {
                                return 0.0;
                            }
                            return tensor.mem(sampled_elem, cochain_elem);
                        },
                        chain,
                        lower_chain,
                        elem);
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
    return coboundary<TagToAddToCochain, CochainTag>(exec_space, coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
