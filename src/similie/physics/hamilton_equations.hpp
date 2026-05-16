// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <similie/physics/magnetostatics/magnetostatics_indices.hpp>

#include <Kokkos_Core.hpp>

namespace similie::physics {

namespace detail {

template <class Hamiltonian>
concept MagnetostaticsHamiltonian = requires(Hamiltonian const& hamiltonian, double value) {
    hamiltonian.dH_dpi0(value);
    hamiltonian.dH_dpi1(value);
    hamiltonian.dH_dpi2(value);
};

template <class Tensor>
concept MagneticFieldLikeTensor = requires(Tensor tensor) {
    tensor.template access_element<magnetostatics::X>();
    tensor.template access_element<magnetostatics::Y>();
    tensor.template access_element<magnetostatics::Z>();
};

template <class Tensor>
concept MagneticInductionLikeTensor = requires(Tensor tensor) {
    tensor.template access_element<magnetostatics::Y, magnetostatics::Z>();
    tensor.template access_element<magnetostatics::X, magnetostatics::Z>();
    tensor.template access_element<magnetostatics::X, magnetostatics::Y>();
};

} // namespace detail

template <class Hamiltonian>
class HamiltonEquations
{
    Hamiltonian m_hamiltonian;

public:
    constexpr explicit HamiltonEquations(Hamiltonian hamiltonian)
        : m_hamiltonian(std::move(hamiltonian))
    {
    }

    [[nodiscard]] constexpr Hamiltonian const& hamiltonian() const
    {
        return m_hamiltonian;
    }

    template <
            class DSpatialMomentumDtTensor,
            class DPotentialDtTensor,
            class SpatialMomentumTensor,
            class PotentialTensor>
    KOKKOS_FUNCTION void run(
            DSpatialMomentumDtTensor dspatial_momentum_dt,
            DPotentialDtTensor dpotential_dt,
            SpatialMomentumTensor spatial_momentum,
            PotentialTensor potential) const
    {
        if constexpr (requires {
                          m_hamiltonian
                                  .run(dspatial_momentum_dt,
                                       dpotential_dt,
                                       spatial_momentum,
                                       potential);
                      }) {
            m_hamiltonian.run(dspatial_momentum_dt, dpotential_dt, spatial_momentum, potential);
        } else if constexpr (
                detail::MagnetostaticsHamiltonian<Hamiltonian>
                && detail::MagneticInductionLikeTensor<DSpatialMomentumDtTensor>
                && detail::MagneticFieldLikeTensor<DPotentialDtTensor>
                && detail::MagneticInductionLikeTensor<SpatialMomentumTensor>) {
            double const bx = spatial_momentum(
                    spatial_momentum
                            .template access_element<magnetostatics::Y, magnetostatics::Z>());
            double const by = -spatial_momentum(
                    spatial_momentum
                            .template access_element<magnetostatics::X, magnetostatics::Z>());
            double const bz = spatial_momentum(
                    spatial_momentum
                            .template access_element<magnetostatics::X, magnetostatics::Y>());

            dpotential_dt(dpotential_dt.template access_element<magnetostatics::X>())
                    = m_hamiltonian.dH_dpi0(bx);
            dpotential_dt(dpotential_dt.template access_element<magnetostatics::Y>())
                    = m_hamiltonian.dH_dpi1(by);
            dpotential_dt(dpotential_dt.template access_element<magnetostatics::Z>())
                    = m_hamiltonian.dH_dpi2(bz);

            dspatial_momentum_dt(
                    dspatial_momentum_dt
                            .template access_element<magnetostatics::Y, magnetostatics::Z>())
                    = 0.0;
            dspatial_momentum_dt(
                    dspatial_momentum_dt
                            .template access_element<magnetostatics::X, magnetostatics::Z>())
                    = 0.0;
            dspatial_momentum_dt(
                    dspatial_momentum_dt
                            .template access_element<magnetostatics::X, magnetostatics::Y>())
                    = 0.0;
        } else {
#ifndef __CUDA_ARCH__
            throw std::logic_error(
                    "HamiltonEquations::run is not implemented for this Hamiltonian/tensor "
                    "combination");
#endif
        }
    }
};

} // namespace similie::physics
