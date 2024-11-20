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

#include "cochain.hpp"
#include "cosimplex.hpp"


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

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
struct CoboundaryTensorType;

template <
        tensor::TensorNatIndex TagToAddToCochain,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
struct CoboundaryTensorType<
        TagToAddToCochain,
        tensor::TensorNaturalIndex<>,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    static_assert(ddc::type_seq_contains_v<
                  ddc::detail::TypeSeq<tensor::TensorNaturalIndex<>>,
                  ddc::detail::TypeSeq<DDim...>>);
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    tensor::TensorNaturalIndex<>,
                    TagToAddToCochain>,
            SupportType,
            MemorySpace>;
};

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorNatIndex NaturalCochainTag,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
struct CoboundaryTensorType<
        TagToAddToCochain,
        NaturalCochainTag,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    static_assert(ddc::type_seq_contains_v<
                  ddc::detail::TypeSeq<NaturalCochainTag>,
                  ddc::detail::TypeSeq<DDim...>>);
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    NaturalCochainTag,
                    tensor::TensorAntisymmetricIndex<TagToAddToCochain, NaturalCochainTag>>,
            SupportType,
            MemorySpace>;
};

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorNatIndex... NaturalCochainTag,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
struct CoboundaryTensorType<
        TagToAddToCochain,
        tensor::TensorAntisymmetricIndex<NaturalCochainTag...>,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    static_assert(ddc::type_seq_contains_v<
                  ddc::detail::TypeSeq<tensor::TensorAntisymmetricIndex<NaturalCochainTag...>>,
                  ddc::detail::TypeSeq<DDim...>>);
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    tensor::TensorAntisymmetricIndex<NaturalCochainTag...>,
                    tensor::TensorAntisymmetricIndex<TagToAddToCochain, NaturalCochainTag...>>,
            SupportType,
            MemorySpace>;
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
using coboundary_t = typename detail::CoboundaryType<CochainType>::type;

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

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,

        misc::Specialization<tensor::Tensor> TensorType,
        class... ODDim>
KOKKOS_FUNCTION coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary(
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    auto antisymmetric_coboundary_tensor = tensor::relabelize_indices_of<
            ddc::detail::TypeSeq<TagToAddToCochain, CochainTag>,
            ddc::detail::TypeSeq<
                    tensor::to_tensor_antisymmetric_index_t<TagToAddToCochain>,
                    tensor::to_tensor_antisymmetric_index_t<CochainTag>>>(coboundary_tensor);
    auto antisymmetric_tensor = tensor::relabelize_indices_of<
            ddc::detail::TypeSeq<TagToAddToCochain, CochainTag>,
            ddc::detail::TypeSeq<
                    tensor::to_tensor_antisymmetric_index_t<TagToAddToCochain>,
                    tensor::to_tensor_antisymmetric_index_t<CochainTag>>>(tensor);

    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            ddc::remove_dims_of<typename misc::convert_type_seq_to_t<
                    tensor::TensorAntisymmetricIndex,
                    ddc::type_seq_remove_t<
                            ddc::type_seq_merge_t<
                                    misc::to_type_seq_t<tensor::to_tensor_antisymmetric_index_t<
                                            TagToAddToCochain>>,
                                    misc::to_type_seq_t<
                                            tensor::to_tensor_antisymmetric_index_t<CochainTag>>>,
                            ddc::detail::TypeSeq<
                                    tensor::TensorNaturalIndex<>,
                                    tensor::TensorAntisymmetricIndex<>>>>>(
                    antisymmetric_coboundary_tensor.domain()),
            [&](auto elem) {
                auto chain = tangent_basis<CochainTag::rank() + 1, ddc::DiscreteDomain<ODDim...>>();
                auto lower_chain
                        = tangent_basis<CochainTag::rank(), ddc::DiscreteDomain<ODDim...>>();
                auto cochain = Cochain(chain, antisymmetric_coboundary_tensor[elem]);
                for (auto i = cochain.begin(); i < cochain.end(); ++i) {
                    sil::exterior::Chain simplex_boundary
                            = boundary(sil::exterior::
                                               Simplex(std::integral_constant<
                                                               std::size_t,
                                                               CochainTag::rank() + 1> {},
                                                       elem,
                                                       (*i).discrete_vector()));
                    std::vector<double> values(simplex_boundary.size());
                    for (auto j = simplex_boundary.begin(); j < simplex_boundary.end(); ++j) {
                        values[std::distance(simplex_boundary.begin(), j)] = antisymmetric_tensor.mem(
                                misc::domain_contains(
                                        antisymmetric_tensor.domain(),
                                        j->discrete_element())
                                        ? j->discrete_element()
                                        : elem, // TODO this is an assumption on boundary condition (free boundary), needs to be generalized
                                ddc::DiscreteElement<
                                        tensor::to_tensor_antisymmetric_index_t<CochainTag>>(
                                        std::distance(
                                                lower_chain.begin(),
                                                std::
                                                        find(lower_chain.begin(),
                                                             lower_chain.end(),
                                                             j->discrete_vector()))));
                    }
                    sil::exterior::Cochain<decltype(simplex_boundary)>
                            cochain_boundary(simplex_boundary, values);
                    antisymmetric_coboundary_tensor.mem(
                            elem,
                            ddc::DiscreteElement<typename misc::convert_type_seq_to_t<
                                    tensor::TensorAntisymmetricIndex,
                                    ddc::type_seq_remove_t<
                                            ddc::type_seq_merge_t<
                                                    misc::to_type_seq_t<
                                                            tensor::to_tensor_antisymmetric_index_t<
                                                                    TagToAddToCochain>>,
                                                    misc::to_type_seq_t<
                                                            tensor::to_tensor_antisymmetric_index_t<
                                                                    CochainTag>>>,
                                            ddc::detail::TypeSeq<
                                                    tensor::TensorNaturalIndex<>,
                                                    tensor::TensorAntisymmetricIndex<>>>>>(
                                    std::distance(cochain.begin(), i)))
                            = cochain_boundary.integrate();
                }
            });

    return coboundary_tensor;
}

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,

        misc::Specialization<tensor::Tensor> TensorType,
        class... ODDim>
KOKKOS_FUNCTION coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> deriv(
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    return coboundary<
            TagToAddToCochain,
            CochainTag,
            TensorType,
            ODDim...>(coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
