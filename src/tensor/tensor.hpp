// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

class Tensor
{
public:
    explicit Tensor();
};

Tensor::Tensor()
{
    printf("Tensor created");
}

} // namespace tensor

} // namespace sil
