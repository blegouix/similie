// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <span>

#include <Kokkos_Core.hpp>

#include <similie/exterior/coboundary.hpp>
#include <similie/exterior/codifferential.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/physics/magnetostatics/linear_magnetic_induction_to_magnetic_field.hpp>
#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>
#include <similie/tensor/tensor.hpp>

namespace similie::physics::magnetostatics {

namespace detail {

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
        sil::exterior::Coboundary<sil::tensor::Covariant<Nu>, MagneticVectorPotentialIndex>::run(
                magnetic_induction,
                evaluator,
                chain,
                lower_chain,
                elem);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 3> forward(
            [[maybe_unused]] std::span<double const, 3> magnetic_vector_potential,
            std::span<double const, 3> dpotential_dx,
            std::span<double const, 3> dpotential_dy,
            std::span<double const, 3> dpotential_dz) const
    {
        return {
                dpotential_dy[2] - dpotential_dz[1],
                dpotential_dz[0] - dpotential_dx[2],
                dpotential_dx[1] - dpotential_dy[0],
        };
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 6> inverse(
            std::span<double const, 3>) const
    {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
};

class MaxwellStressTensorToMagneticInductionAndMagneticField
{
    LinearMagneticInductionToMagneticField m_constitutive_law;

public:
    explicit constexpr MaxwellStressTensorToMagneticInductionAndMagneticField(
            LinearMagneticInductionToMagneticField constitutive_law)
        : m_constitutive_law(constitutive_law)
    {
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 6> inverse(
            std::span<double const, 3> magnetic_induction,
            std::span<double const, 3> magnetic_field) const
    {
        double const half_trace = 0.5
                                  * (magnetic_induction[0] * magnetic_field[0]
                                     + magnetic_induction[1] * magnetic_field[1]
                                     + magnetic_induction[2] * magnetic_field[2]);
        return {
                magnetic_induction[0] * magnetic_field[0] - half_trace,
                magnetic_induction[1] * magnetic_field[1] - half_trace,
                magnetic_induction[2] * magnetic_field[2] - half_trace,
                magnetic_induction[0] * magnetic_field[1],
                magnetic_induction[0] * magnetic_field[2],
                magnetic_induction[1] * magnetic_field[2],
        };
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 6> forward(
            std::span<double const, 3> hodge_star,
            std::span<double const, 3> magnetic_induction) const
    {
        std::array<double, 3> const magnetic_field = {
                m_constitutive_law.forward(hodge_star[0], magnetic_induction[0]),
                m_constitutive_law.forward(hodge_star[1], magnetic_induction[1]),
                m_constitutive_law.forward(hodge_star[2], magnetic_induction[2]),
        };
        return inverse(magnetic_induction, magnetic_field);
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
                PositionType>::run(
                force_density_tensor,
                maxwell_stress_tensor,
                metric,
                position,
                chain,
                lower_chain,
                elem);
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 6> inverse(
            std::span<double const, 3>) const
    {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    [[nodiscard]] KOKKOS_FUNCTION constexpr std::array<double, 3> forward(
            std::span<double const, 6>) const
    {
        return {0.0, 0.0, 0.0};
    }
};

} // namespace similie::physics::magnetostatics
