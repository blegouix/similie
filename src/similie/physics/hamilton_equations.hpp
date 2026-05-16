// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <Kokkos_Core.hpp>

namespace similie::physics {

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
        if constexpr (requires(Hamiltonian const& hamiltonian) {
                          run_hamilton_equations(
                                  hamiltonian,
                                  dspatial_momentum_dt,
                                  dpotential_dt,
                                  spatial_momentum,
                                  potential);
                      }) {
            run_hamilton_equations(
                    m_hamiltonian,
                    dspatial_momentum_dt,
                    dpotential_dt,
                    spatial_momentum,
                    potential);
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
