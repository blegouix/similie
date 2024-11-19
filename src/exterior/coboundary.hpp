// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include "antisymmetric_tensor.hpp"
#include "are_all_same.hpp"
#include "cochain.hpp"
#include "cosimplex.hpp"
#include "domain_contains.hpp"
#include "filled_struct.hpp"
#include "specialization.hpp"
#include "type_seq_conversion.hpp"

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
        misc::Specialization<tensor::TensorAntisymmetricIndex> CochainTag,
        tensor::TensorNatIndex TagToAddToCochain,
        misc::Specialization<tensor::Tensor> TensorType>
struct CoboundaryTensorType;

template <
        tensor::TensorNatIndex... NaturalCochainTag,
        tensor::TensorNatIndex TagToAddToCochain,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
struct CoboundaryTensorType<
        tensor::TensorAntisymmetricIndex<NaturalCochainTag...>,
        TagToAddToCochain,
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
        misc::Specialization<tensor::TensorAntisymmetricIndex> CochainTag,
        tensor::TensorNatIndex TagToAddToCochain,
        misc::Specialization<tensor::Tensor> TensorType>
using coboundary_tensor_t =
        typename detail::CoboundaryTensorType<CochainTag, TagToAddToCochain, TensorType>::type;

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
        misc::Specialization<tensor::TensorAntisymmetricIndex> CochainTag,
        tensor::TensorNatIndex TagToAddToCochain,
        misc::Specialization<tensor::Tensor> AntisymmetricTensorType>
KOKKOS_FUNCTION coboundary_tensor_t<CochainTag, TagToAddToCochain, AntisymmetricTensorType>
coboundary(
        coboundary_tensor_t<CochainTag, TagToAddToCochain, AntisymmetricTensorType>
                coboundary_tensor,
        AntisymmetricTensorType tensor)
{
    ddc::parallel_for_each(
            Kokkos::DefaultHostExecutionSpace(),
            coboundary_tensor.non_indices_domain(),
            [&](auto elem) {
                auto chain = tangent_basis<
                        CochainTag::rank() + 1,
                        typename AntisymmetricTensorType::non_indices_domain_t>();
                auto lower_chain = tangent_basis<
                        CochainTag::rank(),
                        typename AntisymmetricTensorType::non_indices_domain_t>();
                auto cochain = Cochain(chain, coboundary_tensor[elem]);
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
                        values[std::distance(simplex_boundary.begin(), j)] = tensor.mem(
                                misc::domain_contains(tensor.domain(), j->discrete_element())
                                        ? j->discrete_element()
                                        : elem, // TODO this is an assumption on boundary condition (free boundary), needs to be generalized
                                ddc::DiscreteElement<CochainTag>(std::distance(
                                        lower_chain.begin(),
                                        std::
                                                find(lower_chain.begin(),
                                                     lower_chain.end(),
                                                     j->discrete_vector()))));
                    }
                    sil::exterior::Cochain<decltype(simplex_boundary)>
                            cochain_boundary(simplex_boundary, values);
                    coboundary_tensor
                            .mem(elem,
                                 ddc::DiscreteElement<typename misc::convert_type_seq_to_t<
                                         tensor::TensorAntisymmetricIndex,
                                         ddc::type_seq_merge_t<
                                                 ddc::detail::TypeSeq<TagToAddToCochain>,
                                                 misc::to_type_seq_t<CochainTag>>>>(
                                         std::distance(cochain.begin(), i)))
                            = cochain_boundary.integrate();
                }
            });

    return coboundary_tensor;
}

template <
        misc::Specialization<tensor::TensorAntisymmetricIndex> CochainTag,
        tensor::TensorNatIndex TagToAddToCochain,
        misc::Specialization<tensor::Tensor> AntisymmetricTensorType>
KOKKOS_FUNCTION coboundary_tensor_t<CochainTag, TagToAddToCochain, AntisymmetricTensorType> deriv(
        coboundary_tensor_t<CochainTag, TagToAddToCochain, AntisymmetricTensorType>
                coboundary_tensor,
        AntisymmetricTensorType tensor)
{
    return coboundary<
            CochainTag,
            TagToAddToCochain,
            AntisymmetricTensorType>(coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
