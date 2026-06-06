// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>

#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/codifferential.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/tensor.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

template <class... SpatialIndex>
struct Mu : sil::tensor::TensorNaturalIndex<SpatialIndex...>
{
};

template <class... SpatialIndex>
struct Nu : sil::tensor::TensorNaturalIndex<SpatialIndex...>
{
};

template <class... SpatialIndex>
using MagneticVectorPotentialIndex = sil::tensor::Covariant<Mu<SpatialIndex...>>;
template <class... SpatialIndex>
using MagneticFieldIndex = sil::tensor::Covariant<Mu<SpatialIndex...>>;
template <class... SpatialIndex>
using ForceDensityIndex = sil::tensor::Covariant<Mu<SpatialIndex...>>;
template <class... SpatialIndex>
using MagneticInductionIndex = sil::tensor::TensorAntisymmetricIndex<
        sil::tensor::Covariant<Mu<SpatialIndex...>>,
        sil::tensor::Covariant<Nu<SpatialIndex...>>>;
template <class... SpatialIndex>
using MaxwellStressTensorIndex = sil::tensor::TensorSymmetricIndex<
        sil::tensor::Covariant<Mu<SpatialIndex...>>,
        sil::tensor::Covariant<Nu<SpatialIndex...>>>;

namespace detail {

template <class... SpatialIndex>
struct InPlaneNu : sil::tensor::TensorNaturalIndex<SpatialIndex...>
{
};

template <class Elem>
struct ElementSpatialDomain;

template <class... Tags>
struct ElementSpatialDomain<ddc::DiscreteElement<Tags...>>
{
    using type = ddc::DiscreteDomain<Tags...>;
};

} // namespace detail

template <class XIndex, class YIndex, class... OtherSpatialIndex>
class MagneticVectorPotentialToMagneticInduction
{
public:
    template <class Index, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(Elem elem)
    {
        static_assert(
                std::is_same_v<Index, XIndex> || std::is_same_v<Index, YIndex>,
                "Magnetic induction component tag must be one of the two in-plane spatial tags");
        using spatial_domain_type = typename detail::ElementSpatialDomain<Elem>::type;
        using PotentialScalarIndex = sil::tensor::ScalarIndex;
        using CoboundaryOutputIndex = sil::exterior::coboundary_index_t<
                sil::tensor::Covariant<detail::InPlaneNu<XIndex, YIndex, OtherSpatialIndex...>>,
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
                sil::tensor::Covariant<detail::InPlaneNu<XIndex, YIndex, OtherSpatialIndex...>>,
                PotentialScalarIndex>::
                value([](auto, auto) { return 0.0; }, chain, lower_chain, elem, natural_elem);
        if constexpr (std::is_same_v<Index, YIndex>) {
            stencil *= -1.0;
        }
        return stencil;
    }

    template <
            class CoboundaryTensorType,
            class Evaluator,
            class ChainType,
            class LowerChainType,
            class Elem>
    KOKKOS_FUNCTION static void forward(
            CoboundaryTensorType magnetic_induction,
            Evaluator evaluator,
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem)
    {
        sil::exterior::Coboundary<
                sil::tensor::Covariant<Nu<XIndex, YIndex, OtherSpatialIndex...>>,
                MagneticVectorPotentialIndex<XIndex, YIndex, OtherSpatialIndex...>>::
                run(magnetic_induction, evaluator, chain, lower_chain, elem);
    }
};

class MaxwellStressTensorToMagneticInductionAndMagneticField
{
    double m_mu;

public:
    explicit constexpr MaxwellStressTensorToMagneticInductionAndMagneticField(double mu) : m_mu(mu)
    {
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double inverse_value(
            std::span<double const, 3> magnetic_induction,
            std::span<double const, 3> magnetic_field,
            std::size_t component) const
    {
        double const half_trace = 0.5
                                  * (magnetic_induction[0] * magnetic_field[0]
                                     + magnetic_induction[1] * magnetic_field[1]
                                     + magnetic_induction[2] * magnetic_field[2]);
        switch (component) {
        case 0:
            return magnetic_induction[0] * magnetic_field[0] - half_trace;
        case 1:
            return magnetic_induction[1] * magnetic_field[1] - half_trace;
        case 2:
            return magnetic_induction[2] * magnetic_field[2] - half_trace;
        case 3:
            return magnetic_induction[0] * magnetic_field[1];
        case 4:
            return magnetic_induction[0] * magnetic_field[2];
        default:
            return magnetic_induction[1] * magnetic_field[2];
        }
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 6> inverse(
            std::span<double const, 3> magnetic_induction,
            std::span<double const, 3> magnetic_field) const
    {
        return {
                inverse_value(magnetic_induction, magnetic_field, 0),
                inverse_value(magnetic_induction, magnetic_field, 1),
                inverse_value(magnetic_induction, magnetic_field, 2),
                inverse_value(magnetic_induction, magnetic_field, 3),
                inverse_value(magnetic_induction, magnetic_field, 4),
                inverse_value(magnetic_induction, magnetic_field, 5),
        };
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double forward_value(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction,
            std::size_t component) const
    {
        std::array<double, 3> const magnetic_field = {
                hodge_star[0] * magnetic_induction[0] / m_mu,
                hodge_star[1] * magnetic_induction[1] / m_mu,
                hodge_star[2] * magnetic_induction[2] / m_mu,
        };
        return inverse_value(magnetic_induction, magnetic_field, component);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 6> forward(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction) const
    {
        return {
                forward_value(hodge_star, magnetic_induction, 0),
                forward_value(hodge_star, magnetic_induction, 1),
                forward_value(hodge_star, magnetic_induction, 2),
                forward_value(hodge_star, magnetic_induction, 3),
                forward_value(hodge_star, magnetic_induction, 4),
                forward_value(hodge_star, magnetic_induction, 5),
        };
    }
};

template <class... SpatialIndex>
class ForceDensityToMaxwellStressTensor
{
public:
    template <
            sil::tensor::TensorIndex MetricIndex,
            sil::misc::Specialization<sil::tensor::Tensor> CodifferentialTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> TensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MetricType,
            sil::misc::Specialization<sil::tensor::Tensor> PositionType,
            class ChainType,
            class LowerChainType,
            class Elem>
    KOKKOS_FUNCTION static void forward(
            CodifferentialTensorType force_density_tensor,
            TensorType maxwell_stress_tensor,
            MetricType metric,
            PositionType position,
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem)
    {
        sil::exterior::Codifferential<
                MetricIndex,
                sil::tensor::Covariant<Nu<SpatialIndex...>>,
                MaxwellStressTensorIndex<SpatialIndex...>,
                TensorType,
                MetricType,
                PositionType>::
                run(force_density_tensor,
                    maxwell_stress_tensor,
                    metric,
                    position,
                    chain,
                    lower_chain,
                    elem);
    }
};

} // namespace similie::physics::magnetostatics
