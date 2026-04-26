// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

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

#include "cochain.hpp"
#include "cosimplex.hpp"


namespace sil {

namespace exterior {

struct CenteredMirroredBoundary
{
};

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

template <class CDim, class TypeSeq>
struct DiscreteDimensionForTypeSeq;

template <class CDim, class DDim, class... Tail>
struct DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<DDim, Tail...>>
{
    using type = std::conditional_t<
            std::is_same_v<typename DDim::continuous_dimension_type, CDim>,
            DDim,
            typename DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<Tail...>>::type>;
};

template <class CDim>
struct DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<>>
{
    using type = void;
};

template <class CDim, class T>
using discrete_dimension_for_t
        = typename DiscreteDimensionForTypeSeq<CDim, ddc::to_type_seq_t<T>>::type;

template <class DDim, class Vect>
KOKKOS_FUNCTION Vect unit_shift()
{
    Vect shift = misc::filled_struct<Vect>();
    shift.template get<DDim>() = 1;
    return shift;
}

template <class DDim, class Domain, class Elem>
KOKKOS_FUNCTION double centered_step(Domain const& domain, Elem const& elem)
{
    using vect_t = typename Domain::discrete_vector_type;
    vect_t const shift = unit_shift<DDim, vect_t>();
    ddc::DiscreteDomain<DDim> const dim_dom(domain);
    ddc::DiscreteElement<DDim> const dim_elem(elem);
    bool const has_left = dim_elem.uid() != dim_dom.front().uid();
    bool const has_right = dim_elem.uid() != dim_dom.back().uid();

    if (has_left && has_right) {
        return static_cast<double>(
                ddc::coordinate(ddc::DiscreteElement<DDim>(elem + shift))
                - ddc::coordinate(ddc::DiscreteElement<DDim>(elem - shift)));
    } else if (has_right) {
        return 2.
               * static_cast<double>(
                       ddc::coordinate(ddc::DiscreteElement<DDim>(elem + shift))
                       - ddc::coordinate(ddc::DiscreteElement<DDim>(elem)));
    } else {
        return 2.
               * static_cast<double>(
                       ddc::coordinate(ddc::DiscreteElement<DDim>(elem))
                       - ddc::coordinate(ddc::DiscreteElement<DDim>(elem - shift)));
    }
}

template <class DDim, class Domain, class Elem>
KOKKOS_FUNCTION Elem mirrored_left_elem(Domain const& domain, Elem const& elem)
{
    using vect_t = typename Domain::discrete_vector_type;
    vect_t const shift = unit_shift<DDim, vect_t>();
    ddc::DiscreteDomain<DDim> const dim_dom(domain);
    ddc::DiscreteElement<DDim> const dim_elem(elem);
    return dim_elem.uid() != dim_dom.front().uid() ? elem - shift : elem + shift;
}

template <class DDim, class Domain, class Elem>
KOKKOS_FUNCTION Elem mirrored_right_elem(Domain const& domain, Elem const& elem)
{
    using vect_t = typename Domain::discrete_vector_type;
    vect_t const shift = unit_shift<DDim, vect_t>();
    ddc::DiscreteDomain<DDim> const dim_dom(domain);
    ddc::DiscreteElement<DDim> const dim_elem(elem);
    return dim_elem.uid() != dim_dom.back().uid() ? elem + shift : elem - shift;
}

template <
        class NaturalDims,
        class OutTensorType,
        class TensorType,
        class Elem,
        class OutIndex,
        class InIndex>
struct FillCenteredDerivative;

template <
        class Axis,
        class... TailAxes,
        class OutTensorType,
        class TensorType,
        class Elem,
        class OutIndex,
        class InIndex>
struct FillCenteredDerivative<ddc::detail::TypeSeq<Axis, TailAxes...>, OutTensorType, TensorType, Elem, OutIndex, InIndex>
{
    KOKKOS_FUNCTION static void run(OutTensorType out_tensor, TensorType tensor, Elem elem)
    {
        using d_dim_t = discrete_dimension_for_t<Axis, typename TensorType::non_indices_domain_t>;
        static_assert(!std::is_void_v<d_dim_t>);
        constexpr std::size_t comp = ddc::type_seq_rank_v<Axis, typename OutIndex::type_seq_dimensions>;

        auto const left_elem = mirrored_left_elem<d_dim_t>(tensor.non_indices_domain(), elem);
        auto const right_elem = mirrored_right_elem<d_dim_t>(tensor.non_indices_domain(), elem);
        double const denom = centered_step<d_dim_t>(tensor.non_indices_domain(), elem);

        out_tensor.mem(elem, ddc::DiscreteElement<OutIndex>(comp))
                = (tensor.get(right_elem, ddc::DiscreteElement<InIndex>(0))
                   - tensor.get(left_elem, ddc::DiscreteElement<InIndex>(0)))
                  / denom;

        FillCenteredDerivative<
                ddc::detail::TypeSeq<TailAxes...>,
                OutTensorType,
                TensorType,
                Elem,
                OutIndex,
                InIndex>::run(out_tensor, tensor, elem);
    }
};

template <class OutTensorType, class TensorType, class Elem, class OutIndex, class InIndex>
struct FillCenteredDerivative<ddc::detail::TypeSeq<>, OutTensorType, TensorType, Elem, OutIndex, InIndex>
{
    KOKKOS_FUNCTION static void run(
            [[maybe_unused]] OutTensorType out_tensor,
            [[maybe_unused]] TensorType tensor,
            [[maybe_unused]] Elem elem)
    {
    }
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

    // buffer to store the K-chain containing the boundary of each K+1-simplex of the mesh
    ddc::Chunk simplex_boundary_alloc(
            ddc::cartesian_prod_t<
                    ddc::remove_dims_of_t<
                            typename coboundary_tensor_t<
                                    TagToAddToCochain,
                                    CochainTag,
                                    TensorType>::discrete_domain_type,
                            coboundary_index_t<TagToAddToCochain, CochainTag>>,
                    ddc::DiscreteDomain<detail::CoboundaryDummyIndex>>(
                    batch_dom,
                    ddc::DiscreteDomain<detail::CoboundaryDummyIndex>(
                            ddc::DiscreteElement<detail::CoboundaryDummyIndex>(0),
                            ddc::DiscreteVector<detail::CoboundaryDummyIndex>(
                                    2 * (CochainTag::rank() + 1)))),
            ddc::KokkosAllocator<
                    simplex_for_domain_t<
                            CochainTag::rank(),
                            ddc::remove_dims_of_t<
                                    typename coboundary_tensor_t<
                                            TagToAddToCochain,
                                            CochainTag,
                                            TensorType>::discrete_domain_type,
                                    coboundary_index_t<TagToAddToCochain, CochainTag>>>,
                    typename ExecSpace::memory_space>());
    ddc::ChunkSpan simplex_boundary(simplex_boundary_alloc);

    // buffer to store the values of the K-cochain on the boundary of each K+1-cosimplex of the mesh
    ddc::Chunk boundary_values_alloc(
            ddc::cartesian_prod_t<
                    ddc::remove_dims_of_t<
                            typename coboundary_tensor_t<
                                    TagToAddToCochain,
                                    CochainTag,
                                    TensorType>::discrete_domain_type,
                            coboundary_index_t<TagToAddToCochain, CochainTag>>,
                    ddc::DiscreteDomain<detail::CoboundaryDummyIndex>>(
                    batch_dom,
                    ddc::DiscreteDomain<detail::CoboundaryDummyIndex>(
                            ddc::DiscreteElement<detail::CoboundaryDummyIndex>(0),
                            ddc::DiscreteVector<detail::CoboundaryDummyIndex>(
                                    2 * (CochainTag::rank() + 1)))),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan boundary_values(boundary_values_alloc);

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
                // declare a K+1-cochain storing the K+1-cosimplices of the output cochain for the current tangent space and iterate over them
                auto cochain = Cochain(chain, coboundary_tensor[elem]);
                for (auto i = cochain.begin(); i < cochain.end(); ++i) {
                    // extract the K+1-simplex from the current K+1-cosimplex (this is not absolutly trivial because the cochain is based on a LocalChain)
                    Simplex simplex = Simplex(
                            std::integral_constant<std::size_t, CochainTag::rank() + 1> {},
                            elem,
                            (*i).discrete_vector());
                    // compute its boundary as a K-chain in the simplex_boundary buffer
                    Chain boundary_chain
                            = boundary(simplex_boundary[elem].allocation_kokkos_view(), simplex);
                    // iterate over every K-simplex forming the boundary
                    for (auto j = boundary_chain.begin(); j < boundary_chain.end(); ++j) {
                        // extract from the input K-cochain the values associated to every K-simplex of the boundary and fill the boundary_values buffer
                        boundary_values[elem].allocation_kokkos_view()(
                                Kokkos::Experimental::distance(boundary_chain.begin(), j))
                                = tensor.mem(
                                        misc::domain_contains(
                                                tensor.domain(),
                                                (*j).discrete_element())
                                                ? (*j).discrete_element()
                                                : elem, // TODO this is an assumption on boundary condition (free boundary), needs to be generalized
                                        ddc::DiscreteElement<CochainTag>(
                                                Kokkos::Experimental::distance(
                                                        lower_chain.begin(),
                                                        misc::detail::
                                                                find(lower_chain.begin(),
                                                                     lower_chain.end(),
                                                                     (*j).discrete_vector()))));
                    }
                    // build the cochain of the boundary
                    Cochain cochain_boundary(
                            boundary_chain,
                            boundary_values[elem].allocation_kokkos_view());
                    // integrate over the cochain forming the boundary to compute the coboundary
                    // (*i).value() = cochain_boundary.integrate(); // Cannot be used because CochainIterator::operator* does not return a reference
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

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
    requires(CochainTag::rank() == 0 && TagToAddToCochain::rank() == 1)
coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> deriv(
        ExecSpace const& exec_space,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor,
        CenteredMirroredBoundary)
{
    using out_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename decltype(coboundary_tensor)::indices_domain_t>>;
    using in_index_t = ddc::type_seq_element_t<
            0,
            ddc::to_type_seq_t<typename TensorType::indices_domain_t>>;
    static_assert(out_index_t::rank() == 1);
    static_assert(in_index_t::rank() == 0);

    ddc::parallel_for_each(
            "similie_compute_centered_mirrored_deriv",
            exec_space,
            coboundary_tensor.non_indices_domain(),
            KOKKOS_LAMBDA(typename decltype(coboundary_tensor)::non_indices_domain_t::discrete_element_type elem) {
                detail::FillCenteredDerivative<
                        typename out_index_t::type_seq_dimensions,
                        decltype(coboundary_tensor),
                        TensorType,
                        decltype(elem),
                        out_index_t,
                        in_index_t>::run(coboundary_tensor, tensor, elem);
            });

    return coboundary_tensor;
}

} // namespace exterior

} // namespace sil
