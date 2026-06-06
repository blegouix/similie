// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <similie/exterior/coboundary.hpp>
#include <similie/tensor/tensor.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

template <class... SpatialIndex>
using ForceDensityIndex
        = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>;

namespace detail {

template <class Elem>
struct ElementSpatialDomain;

template <class... Tags>
struct ElementSpatialDomain<ddc::DiscreteElement<Tags...>>
{
    using type = ddc::DiscreteDomain<Tags...>;
};

} // namespace detail

template <class... SpatialIndex>
class MagneticVectorPotentialToMagneticInduction
{
public:
    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(Elem elem)
    {
        static_assert(sizeof...(SpatialIndex) >= 2);
        using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
        using XIndex = ddc::type_seq_element_t<0, SpatialIndexSeq>;
        using YIndex = ddc::type_seq_element_t<1, SpatialIndexSeq>;
        static_assert(
                std::is_same_v<Index, XIndex> || std::is_same_v<Index, YIndex>,
                "Magnetic induction component tag must be one of the two in-plane spatial tags");
        using spatial_domain_type = typename detail::ElementSpatialDomain<Elem>::type;
        using PotentialScalarIndex = sil::tensor::ScalarIndex;
        using CoboundaryOutputIndex = sil::exterior::coboundary_index_t<
                sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>,
                PotentialScalarIndex>;
        [[maybe_unused]] sil::tensor::TensorAccessor<CoboundaryOutputIndex> accessor;
        auto const natural_elem = [&]() {
            if constexpr (std::is_same_v<Index, XIndex>) {
                return accessor.template access_element<YIndex>();
            } else {
                return accessor.template access_element<XIndex>();
            }
        }();
        auto const chain = sil::exterior::tangent_basis<1, spatial_domain_type>(elem);
        auto const lower_chain = sil::exterior::tangent_basis<0, spatial_domain_type>(elem);

        auto stencil = sil::exterior::Coboundary<
                sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>,
                PotentialScalarIndex>::
                value([](auto, auto) { return 0.0; }, chain, lower_chain, elem, natural_elem);
        if constexpr (std::is_same_v<Index, YIndex>) {
            stencil *= -1.0;
        }
        return stencil;
    }
};

} // namespace similie::physics::magnetostatics
