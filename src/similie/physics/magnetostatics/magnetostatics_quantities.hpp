// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <similie/misc/specialization.hpp>
#include <similie/physics/magnetostatics/linear_magnetostatics.hpp>
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

template <
        sil::misc::Specialization<sil::tensor::Tensor> DSpatialMomentumDtTensorType,
        sil::misc::Specialization<sil::tensor::Tensor> DPotentialDtTensorType,
        sil::misc::Specialization<sil::tensor::Tensor> SpatialMomentumTensorType,
        class PotentialTensorType>
KOKKOS_FUNCTION void run_hamilton_equations(
        LinearMagnetostaticsHamiltonian const& hamiltonian,
        DSpatialMomentumDtTensorType dspatial_momentum_dt,
        DPotentialDtTensorType dpotential_dt,
        SpatialMomentumTensorType spatial_momentum,
        PotentialTensorType)
{
    double const bx = spatial_momentum(spatial_momentum.template access_element<Y, Z>());
    double const by = -spatial_momentum(spatial_momentum.template access_element<X, Z>());
    double const bz = spatial_momentum(spatial_momentum.template access_element<X, Y>());

    dpotential_dt(dpotential_dt.template access_element<X>()) = hamiltonian.dH_dpi0(bx);
    dpotential_dt(dpotential_dt.template access_element<Y>()) = hamiltonian.dH_dpi1(by);
    dpotential_dt(dpotential_dt.template access_element<Z>()) = hamiltonian.dH_dpi2(bz);

    dspatial_momentum_dt(dspatial_momentum_dt.template access_element<Y, Z>()) = 0.0;
    dspatial_momentum_dt(dspatial_momentum_dt.template access_element<X, Z>()) = 0.0;
    dspatial_momentum_dt(dspatial_momentum_dt.template access_element<X, Y>()) = 0.0;
}

class MagneticVectorPotentialToMagneticInduction
{
public:
    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticVectorPotentialTensorType>
    KOKKOS_FUNCTION void forward(
            MagneticInductionTensorType magnetic_induction,
            MagneticVectorPotentialTensorType potential,
            double dAy_dz,
            double dAz_dy,
            double dAz_dx,
            double dAx_dz,
            double dAx_dy,
            double dAy_dx) const
    {
        magnetic_induction(magnetic_induction.template access_element<Y, Z>()) = dAz_dy - dAy_dz;
        magnetic_induction(magnetic_induction.template access_element<X, Z>()) = dAx_dz - dAz_dx;
        magnetic_induction(magnetic_induction.template access_element<X, Y>()) = dAy_dx - dAx_dy;
    }

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticVectorPotentialTensorType>
    KOKKOS_FUNCTION void inverse(MagneticVectorPotentialTensorType, MagneticInductionTensorType)
            const
    {
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

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MaxwellStressTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticFieldTensorType>
    KOKKOS_FUNCTION void inverse(
            MaxwellStressTensorType maxwell_stress_tensor,
            MagneticInductionTensorType magnetic_induction,
            MagneticFieldTensorType magnetic_field) const
    {
        double const bx = magnetic_induction(magnetic_induction.template access_element<Y, Z>());
        double const by = -magnetic_induction(magnetic_induction.template access_element<X, Z>());
        double const bz = magnetic_induction(magnetic_induction.template access_element<X, Y>());

        double const hx = magnetic_field(magnetic_field.template access_element<X>());
        double const hy = magnetic_field(magnetic_field.template access_element<Y>());
        double const hz = magnetic_field(magnetic_field.template access_element<Z>());

        double const half_trace = 0.5 * (bx * hx + by * hy + bz * hz);

        maxwell_stress_tensor(maxwell_stress_tensor.template access_element<X, X>())
                = bx * hx - half_trace;
        maxwell_stress_tensor(maxwell_stress_tensor.template access_element<Y, Y>())
                = by * hy - half_trace;
        maxwell_stress_tensor(maxwell_stress_tensor.template access_element<Z, Z>())
                = bz * hz - half_trace;
        maxwell_stress_tensor(maxwell_stress_tensor.template access_element<X, Y>()) = bx * hy;
        maxwell_stress_tensor(maxwell_stress_tensor.template access_element<X, Z>()) = bx * hz;
        maxwell_stress_tensor(maxwell_stress_tensor.template access_element<Y, Z>()) = by * hz;
    }

    template <
            sil::misc::Specialization<sil::tensor::Tensor> MagneticInductionTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MagneticFieldTensorType,
            sil::misc::Specialization<sil::tensor::Tensor> MaxwellStressTensorType>
    KOKKOS_FUNCTION void forward(
            MagneticFieldTensorType magnetic_field,
            MagneticInductionTensorType magnetic_induction,
            MaxwellStressTensorType maxwell_stress_tensor) const
    {
        m_constitutive_law.forward(magnetic_field, magnetic_induction);
        inverse(maxwell_stress_tensor, magnetic_induction, magnetic_field);
    }
};

class ForceDensityToMaxwellStressTensor
{
public:
    template <sil::misc::Specialization<sil::tensor::Tensor> ForceDensityTensorType>
    KOKKOS_FUNCTION void inverse(ForceDensityTensorType) const
    {
    }

    template <sil::misc::Specialization<sil::tensor::Tensor> MaxwellStressTensorType>
    KOKKOS_FUNCTION void forward(MaxwellStressTensorType) const
    {
    }
};

} // namespace similie::physics::magnetostatics
