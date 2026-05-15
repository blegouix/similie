// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <similie/tensor/tensor.hpp>

namespace similie::physics::magnetostatics {

struct X
{
    static constexpr bool PERIODIC = false;
};

struct Y
{
    static constexpr bool PERIODIC = false;
};

struct Z
{
    static constexpr bool PERIODIC = false;
};

struct Mu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<X, Y, Z>
{
};

using MagneticVectorPotentialIndex = sil::tensor::Covariant<Mu>;
using MagneticFieldIndex = sil::tensor::Covariant<Mu>;
using ForceDensityIndex = sil::tensor::Covariant<Mu>;
using MagneticInductionIndex = sil::tensor::
        TensorAntisymmetricIndex<sil::tensor::Covariant<Mu>, sil::tensor::Covariant<Nu>>;
using MaxwellStressTensorIndex
        = sil::tensor::TensorSymmetricIndex<sil::tensor::Covariant<Mu>, sil::tensor::Covariant<Nu>>;

} // namespace similie::physics::magnetostatics
