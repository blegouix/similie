// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>
#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/reduction_and_reconstruction.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::elasticity {

namespace detail {

template <class Elem>
struct ElementSpatialDomain;

template <class... Tags>
struct ElementSpatialDomain<ddc::DiscreteElement<Tags...>>
{
    using type = ddc::DiscreteDomain<Tags...>;
};

} // namespace detail

template <std::size_t I, std::size_t J>
struct StrainTensorIndex
{
    static constexpr std::size_t FIRST = I;
    static constexpr std::size_t SECOND = J;
};

using StrainXX = StrainTensorIndex<0, 0>;
using StrainXY = StrainTensorIndex<0, 1>;
using StrainYY = StrainTensorIndex<1, 1>;

struct Strain2D
{
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;

    template <class Index>
    [[nodiscard]] KOKKOS_FUNCTION constexpr double get() const
    {
        if constexpr (std::is_same_v<Index, StrainXX>) {
            return xx;
        } else if constexpr (std::is_same_v<Index, StrainXY>) {
            return xy;
        } else if constexpr (std::is_same_v<Index, StrainYY>) {
            return yy;
        } else {
            static_assert(
                    std::is_same_v<Index, StrainXX> || std::is_same_v<Index, StrainYY>
                            || std::is_same_v<Index, StrainXY>,
                    "unsupported elasticity strain component index");
        }
    }
};

struct DisplacementToStrain
{
    [[nodiscard]] KOKKOS_FUNCTION static constexpr Strain2D from_gradient(
            double dux_dx,
            double duy_dy,
            double dux_dy,
            double duy_dx)
    {
        return {
                .xx = dux_dx,
                .yy = duy_dy,
                .xy = 0.5 * (dux_dy + duy_dx),
        };
    }

    template <class StrainIndex, class DisplacementComponent, class... SpatialIndex, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(
            Elem elem,
            double inverse_spacing_x,
            double inverse_spacing_y)
    {
        using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
        using X = ddc::type_seq_element_t<0, SpatialIndexSeq>;
        using Y = ddc::type_seq_element_t<1, SpatialIndexSeq>;

        if constexpr (std::is_same_v<StrainIndex, StrainXX>) {
            auto stencil = derivative_stencil_<X, SpatialIndex...>(elem);
            if constexpr (std::is_same_v<DisplacementComponent, X>) {
                stencil *= inverse_spacing_x;
            } else {
                stencil *= 0.0;
            }
            return stencil;
        } else if constexpr (std::is_same_v<StrainIndex, StrainYY>) {
            auto stencil = derivative_stencil_<Y, SpatialIndex...>(elem);
            if constexpr (std::is_same_v<DisplacementComponent, Y>) {
                stencil *= inverse_spacing_y;
            } else {
                stencil *= 0.0;
            }
            return stencil;
        } else if constexpr (std::is_same_v<StrainIndex, StrainXY>) {
            if constexpr (std::is_same_v<DisplacementComponent, X>) {
                auto stencil = derivative_stencil_<Y, SpatialIndex...>(elem);
                stencil *= 0.5 * inverse_spacing_y;
                return stencil;
            } else {
                auto stencil = derivative_stencil_<X, SpatialIndex...>(elem);
                if constexpr (std::is_same_v<DisplacementComponent, Y>) {
                    stencil *= 0.5 * inverse_spacing_x;
                } else {
                    stencil *= 0.0;
                }
                return stencil;
            }
        } else {
            static_assert(
                    std::is_same_v<StrainIndex, StrainXX> || std::is_same_v<StrainIndex, StrainYY>
                            || std::is_same_v<StrainIndex, StrainXY>,
                    "unsupported elasticity strain component index");
        }
    }

    template <
            class StrainIndex,
            class DisplacementComponent,
            class... SpatialIndex,
            class Elem,
            class PositionType>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(Elem elem, PositionType position)
    {
        using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
        using X = ddc::type_seq_element_t<0, SpatialIndexSeq>;
        using Y = ddc::type_seq_element_t<1, SpatialIndexSeq>;

        if constexpr (std::is_same_v<StrainIndex, StrainXX>) {
            auto stencil = reconstructed_derivative_stencil_<X, X, SpatialIndex...>(
                    elem, position);
            if constexpr (!std::is_same_v<DisplacementComponent, X>) {
                stencil *= 0.0;
            }
            return stencil;
        } else if constexpr (std::is_same_v<StrainIndex, StrainYY>) {
            auto stencil = reconstructed_derivative_stencil_<Y, Y, SpatialIndex...>(
                    elem, position);
            if constexpr (!std::is_same_v<DisplacementComponent, Y>) {
                stencil *= 0.0;
            }
            return stencil;
        } else if constexpr (std::is_same_v<StrainIndex, StrainXY>) {
            if constexpr (std::is_same_v<DisplacementComponent, X>) {
                auto stencil = reconstructed_derivative_stencil_<Y, Y, SpatialIndex...>(
                        elem, position);
                stencil *= 0.5;
                return stencil;
            } else {
                auto stencil = reconstructed_derivative_stencil_<X, X, SpatialIndex...>(
                        elem, position);
                if constexpr (std::is_same_v<DisplacementComponent, Y>) {
                    stencil *= 0.5;
                } else {
                    stencil *= 0.0;
                }
                return stencil;
            }
        } else {
            static_assert(
                    std::is_same_v<StrainIndex, StrainXX> || std::is_same_v<StrainIndex, StrainYY>
                            || std::is_same_v<StrainIndex, StrainXY>,
                    "unsupported elasticity strain component index");
        }
    }

private:
    template <class DerivativeIndex, class... SpatialIndex, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto derivative_stencil_(Elem elem)
    {
        using spatial_domain_type = typename detail::ElementSpatialDomain<Elem>::type;
        using DerivativeTensorIndex =
                sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>;
        using DisplacementScalarIndex = sil::tensor::ScalarIndex;

        auto const chain = sil::exterior::tangent_basis<1, spatial_domain_type>(elem);
        auto const lower_chain = sil::exterior::tangent_basis<0, spatial_domain_type>(elem);
        [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeTensorIndex> accessor;
        return sil::exterior::Coboundary<DerivativeTensorIndex, DisplacementScalarIndex>::value(
                sil::exterior::detail::IdentityStencilEvaluator {},
                chain,
                lower_chain,
                elem,
                accessor.template access_element<DerivativeIndex>());
    }

    template <
            class ReducedDerivativeIndex,
            class ReconstructedDerivativeIndex,
            class... SpatialIndex,
            class Elem,
            class PositionType>
    [[nodiscard]] KOKKOS_FUNCTION static auto reconstructed_derivative_stencil_(
            Elem elem,
            PositionType position)
    {
        using SpatialIndexSeq = ddc::detail::TypeSeq<SpatialIndex...>;
        using GradientFormIndex =
                sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<SpatialIndex...>>;
        using GradientIndexSeq = sil::tensor::upper_t<
                ddc::to_type_seq_t<sil::tensor::natural_domain_t<GradientFormIndex>>>;
        using ReconstructionType =
                sil::exterior::Reconstruction<GradientIndexSeq, PositionType, Elem>;
        using ReconstructionAccessor = sil::tensor::tensor_accessor_for_domain_t<
                sil::exterior::reconstruction_domain_t<GradientIndexSeq>>;
        using ReconstructionNaturalElem =
                typename ReconstructionAccessor::natural_domain_t::discrete_element_type;

        auto stencil = derivative_stencil_<ReducedDerivativeIndex, SpatialIndex...>(elem);
        constexpr std::size_t source_id
                = ddc::type_seq_rank_v<ReducedDerivativeIndex, SpatialIndexSeq>;
        constexpr std::size_t target_id
                = ddc::type_seq_rank_v<ReconstructedDerivativeIndex, SpatialIndexSeq>;
        auto const reconstruction_natural_elem =
                sil::exterior::detail::natural_elem_from_flat_ids<ReconstructionNaturalElem>(
                        std::array<std::size_t, 2> {source_id, target_id});
        double const reconstruction_coefficient
                = ReconstructionType::value(position, elem, reconstruction_natural_elem);
        stencil *= reconstruction_coefficient;
        return stencil;
    }
};

struct CauchyStress2D
{
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;

    [[nodiscard]] KOKKOS_FUNCTION double von_mises() const
    {
        return Kokkos::sqrt(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy);
    }
};

} // namespace similie::physics::elasticity
