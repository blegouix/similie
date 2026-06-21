// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <similie/exterior/coboundary.hpp>
#include <similie/tensor/tensor.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

struct OrthogonalPlaneIndex
{
    static constexpr bool PERIODIC = false;
};

template <class... SpatialIndex>
using ForceDensityIndex = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>;

template <class... SpatialIndex>
concept TwoDimensional = sizeof...(SpatialIndex) == 2;

template <class... SpatialIndex>
concept ThreeDimensional = sizeof...(SpatialIndex) == 3;

namespace detail {

template <class... SpatialIndex>
struct CoboundarySpatialIndex : sil::tensor::TensorNaturalIndex<SpatialIndex...>
{
};

template <class... SpatialIndex>
struct VectorPotentialSpatialIndex : sil::tensor::TensorNaturalIndex<SpatialIndex...>
{
};

template <class Elem>
struct ElementSpatialDomain;

template <class... Tags>
struct ElementSpatialDomain<ddc::DiscreteElement<Tags...>>
{
    using type = ddc::DiscreteDomain<Tags...>;
};

template <class Elem, class TensorIndex>
using local_scalar_stencil_t = sil::exterior::detail::
        local_operator_value_t<typename ElementSpatialDomain<Elem>::type, TensorIndex>;

template <class CochainIndex, class VectorStencil, class Elem>
[[nodiscard]] KOKKOS_FUNCTION auto project_to_scalar_potential_stencil(
        VectorStencil vector_stencil,
        Elem elem,
        ddc::DiscreteElement<CochainIndex> selected_potential_component)
{
    using spatial_domain_type = typename ElementSpatialDomain<Elem>::type;
    using ScalarPotentialIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;

    auto projected = sil::exterior::detail::
            make_local_operator_value_tensor<Kokkos::HostSpace, ScalarPotentialIndex>(
                    vector_stencil.non_indices_domain().front());
    ddc::device_for_each(projected.domain(), [&](auto projected_elem) {
        auto const spatial_elem =
                typename spatial_domain_type::discrete_element_type(projected_elem);
        projected.mem(projected_elem)
                = vector_stencil.get(spatial_elem, selected_potential_component);
    });
    static_cast<void>(elem);
    return projected;
}

} // namespace detail

template <class... SpatialIndex>
class MagneticVectorPotentialToMagneticInduction;

template <class... SpatialIndex>
    requires TwoDimensional<SpatialIndex...>
class MagneticVectorPotentialToMagneticInduction<SpatialIndex...>
{
    using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
    using PotentialScalarIndex = sil::tensor::ScalarIndex;
    using MagneticInductionIndex = sil::exterior::coboundary_index_t<
            sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>,
            PotentialScalarIndex>;
    using OrthogonalPotentialComponent = sil::tensor::TensorNaturalIndex<OrthogonalPlaneIndex>;

    template <class Index>
    [[nodiscard]] KOKKOS_FUNCTION static constexpr auto magnetic_induction_component()
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<MagneticInductionIndex> accessor;
        static_cast<void>(sizeof(OrthogonalPotentialComponent));
        if constexpr (std::is_same_v<Index, ddc::type_seq_element_t<0, SpatialIndexSeq>>) {
            return accessor.template access_element<ddc::type_seq_element_t<1, SpatialIndexSeq>>();
        } else {
            return accessor.template access_element<ddc::type_seq_element_t<0, SpatialIndexSeq>>();
        }
    }

public:
    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(Elem elem)
    {
        static_assert(
                std::is_same_v<Index, ddc::type_seq_element_t<0, SpatialIndexSeq>>
                        || std::is_same_v<Index, ddc::type_seq_element_t<1, SpatialIndexSeq>>,
                "2D magnetic induction component tag must be one of the spatial tags");
        using spatial_domain_type = typename detail::ElementSpatialDomain<Elem>::type;
        auto const chain = sil::exterior::tangent_basis<1, spatial_domain_type>(elem);
        auto const lower_chain = sil::exterior::tangent_basis<0, spatial_domain_type>(elem);

        auto stencil = sil::exterior::Coboundary<
                sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>,
                PotentialScalarIndex>::
                value(sil::exterior::detail::IdentityStencilEvaluator {},
                      chain,
                      lower_chain,
                      elem,
                      magnetic_induction_component<Index>());
        if constexpr (std::is_same_v<Index, ddc::type_seq_element_t<1, SpatialIndexSeq>>) {
            stencil *= -1.0;
        }
        return stencil;
    }
};

template <class... SpatialIndex>
    requires ThreeDimensional<SpatialIndex...>
class MagneticVectorPotentialToMagneticInduction<SpatialIndex...>
{
    using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
    using SpatialNaturalIndex = sil::tensor::TensorNaturalIndex<SpatialIndex...>;
    using CoboundaryIndex = sil::tensor::Contravariant<SpatialNaturalIndex>;
    using VectorPotentialIndex = sil::tensor::Covariant<SpatialNaturalIndex>;
    using MagneticInductionIndex
            = sil::exterior::coboundary_index_t<CoboundaryIndex, VectorPotentialIndex>;

    template <class Index>
    [[nodiscard]] KOKKOS_FUNCTION static constexpr auto magnetic_induction_component()
    {
        [[maybe_unused]] sil::tensor::TensorAccessor<MagneticInductionIndex> accessor;
        if constexpr (std::is_same_v<Index, ddc::type_seq_element_t<0, SpatialIndexSeq>>) {
            return accessor.template access_element<
                    ddc::type_seq_element_t<1, SpatialIndexSeq>,
                    ddc::type_seq_element_t<2, SpatialIndexSeq>>();
        } else if constexpr (std::is_same_v<Index, ddc::type_seq_element_t<1, SpatialIndexSeq>>) {
            return accessor.template access_element<
                    ddc::type_seq_element_t<0, SpatialIndexSeq>,
                    ddc::type_seq_element_t<2, SpatialIndexSeq>>();
        } else {
            return accessor.template access_element<
                    ddc::type_seq_element_t<0, SpatialIndexSeq>,
                    ddc::type_seq_element_t<1, SpatialIndexSeq>>();
        }
    }

public:
    using vector_potential_index = VectorPotentialIndex;
    using magnetic_induction_index = MagneticInductionIndex;

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_vector_value(Elem elem)
    {
        static_assert(
                std::is_same_v<Index, ddc::type_seq_element_t<0, SpatialIndexSeq>>
                        || std::is_same_v<Index, ddc::type_seq_element_t<1, SpatialIndexSeq>>
                        || std::is_same_v<Index, ddc::type_seq_element_t<2, SpatialIndexSeq>>,
                "3D magnetic induction component tag must be one of the spatial tags");
        using spatial_domain_type = typename detail::ElementSpatialDomain<Elem>::type;
        auto const chain = sil::exterior::tangent_basis<2, spatial_domain_type>(elem);
        auto const lower_chain = sil::exterior::tangent_basis<1, spatial_domain_type>(elem);

        auto const output_component = magnetic_induction_component<Index>();
        auto stencil = sil::exterior::Coboundary<CoboundaryIndex, VectorPotentialIndex>::
                value(sil::exterior::detail::IdentityStencilEvaluator {},
                      chain,
                      lower_chain,
                      elem,
                      output_component);
        if constexpr (std::is_same_v<Index, ddc::type_seq_element_t<1, SpatialIndexSeq>>) {
            stencil *= -1.0;
        }
        return stencil;
    }

    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(Elem elem)
    {
        auto stencil = forward_vector_value<Index>(elem);
        [[maybe_unused]] sil::tensor::TensorAccessor<VectorPotentialIndex> potential_accessor;
        auto projected = detail::project_to_scalar_potential_stencil(
                stencil,
                elem,
                potential_accessor
                        .template access_element<ddc::type_seq_element_t<2, SpatialIndexSeq>>());
        if constexpr (std::is_same_v<Index, ddc::type_seq_element_t<2, SpatialIndexSeq>>) {
            projected *= 0.0;
        }
        return projected;
    }
};

} // namespace similie::physics::magnetostatics
