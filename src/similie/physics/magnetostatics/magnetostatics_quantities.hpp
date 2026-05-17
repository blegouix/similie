// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <span>

#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/codifferential.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/physics/magnetostatics/linear_magnetic_induction_to_magnetic_field.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/tensor/tensor.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics::magnetostatics {

namespace detail {

struct InPlaneNu : sil::tensor::TensorNaturalIndex<X, Y>
{
};

template <class Quantity>
struct QuantityValueFromPotential
{
    template <std::size_t I, class ChainType, class LowerChainType, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION constexpr auto operator()(
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem) const
    {
        return Quantity::template forward_value<I>(chain, lower_chain, elem);
    }
};

template <class TensorIndex>
KOKKOS_FUNCTION auto make_local_tensor(std::array<double, TensorIndex::access_size()>& storage)
{
    [[maybe_unused]] sil::tensor::TensorAccessor<TensorIndex> accessor;
    ddc::ChunkSpan<
            double,
            ddc::DiscreteDomain<TensorIndex>,
            Kokkos::layout_right,
            Kokkos::HostSpace>
            span(storage.data(), accessor.domain());
    return sil::tensor::Tensor(span);
}

} // namespace detail

class MagneticVectorPotentialToMagneticInduction
{
public:
    template <std::size_t I, class ChainType, class LowerChainType, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto forward_value(
            ChainType chain,
            LowerChainType lower_chain,
            Elem elem)
    {
        static_assert(I < 2);
        using PotentialScalarIndex = sil::tensor::ScalarIndex;
        using CoboundaryOutputIndex = sil::exterior::
                coboundary_index_t<sil::tensor::Covariant<detail::InPlaneNu>, PotentialScalarIndex>;
        [[maybe_unused]] sil::tensor::TensorAccessor<CoboundaryOutputIndex> accessor;
        auto const natural_elem = [&]() {
            if constexpr (I == 0) {
                return accessor.template access_element<Y>();
            } else {
                return accessor.template access_element<X>();
            }
        }();

        auto stencil = sil::exterior::Coboundary<
                sil::tensor::Covariant<detail::InPlaneNu>,
                PotentialScalarIndex>::
                value([](auto, auto) { return 0.0; }, chain, lower_chain, elem, natural_elem);
        if constexpr (I == 1) {
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
        sil::exterior::Coboundary<sil::tensor::Covariant<Nu>, MagneticVectorPotentialIndex>::
                run(magnetic_induction, evaluator, chain, lower_chain, elem);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr double forward_value(
            [[maybe_unused]] std::span<double const, 3> magnetic_vector_potential,
            std::span<double const, 3> dpotential_dx,
            std::span<double const, 3> dpotential_dy,
            std::span<double const, 3> dpotential_dz,
            std::size_t component) const
    {
        switch (component) {
        case 0:
            return dpotential_dy[2] - dpotential_dz[1];
        case 1:
            return dpotential_dz[0] - dpotential_dx[2];
        default:
            return dpotential_dx[1] - dpotential_dy[0];
        }
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 3> forward(
            [[maybe_unused]] std::span<double const, 3> magnetic_vector_potential,
            std::span<double const, 3> dpotential_dx,
            std::span<double const, 3> dpotential_dy,
            std::span<double const, 3> dpotential_dz) const
    {
        return {
                forward_value(
                        magnetic_vector_potential,
                        dpotential_dx,
                        dpotential_dy,
                        dpotential_dz,
                        0),
                forward_value(
                        magnetic_vector_potential,
                        dpotential_dx,
                        dpotential_dy,
                        dpotential_dz,
                        1),
                forward_value(
                        magnetic_vector_potential,
                        dpotential_dx,
                        dpotential_dy,
                        dpotential_dz,
                        2),
        };
    }
};

using MagneticInductionValueFromPotential
        = detail::QuantityValueFromPotential<MagneticVectorPotentialToMagneticInduction>;

class MaxwellStressTensorToMagneticInductionAndMagneticField
{
    LinearMagneticInductionToMagneticField m_constitutive_law;

public:
    explicit constexpr MaxwellStressTensorToMagneticInductionAndMagneticField(
            LinearMagneticInductionToMagneticField constitutive_law)
        : m_constitutive_law(constitutive_law)
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
                m_constitutive_law.forward(hodge_star[0], magnetic_induction[0]),
                m_constitutive_law.forward(hodge_star[1], magnetic_induction[1]),
                m_constitutive_law.forward(hodge_star[2], magnetic_induction[2]),
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
                sil::tensor::Covariant<Nu>,
                MaxwellStressTensorIndex,
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
