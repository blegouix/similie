// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

#include <Kokkos_Core.hpp>

namespace similie::physics {

template <class Hamiltonian>
class DeDonderWeylEquations
{
    Hamiltonian m_hamiltonian;

public:
    constexpr explicit DeDonderWeylEquations(Hamiltonian hamiltonian)
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
            DSpatialMomentumDtTensor,
            DPotentialDtTensor,
            SpatialMomentumTensor,
            PotentialTensor) const
    {
#ifndef __CUDA_ARCH__
        throw std::logic_error(
                "DeDonderWeylEquations::run is a placeholder: relativistic equation support is not "
                "implemented yet");
#endif
    }
};

} // namespace similie::physics
