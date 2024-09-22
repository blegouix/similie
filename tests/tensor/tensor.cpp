// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "tensor.hpp"

struct Mu : sil::tensor::TensorIndex
{
    static const std::size_t s_dim_size;
};
std::size_t const Mu::s_dim_size = 3;

struct Nu : sil::tensor::TensorIndex
{
    static const std::size_t s_dim_size;
};
std::size_t const Nu::s_dim_size = 4;


TEST(Tensor, Init)
{
    sil::tensor::TensorDomain<Mu, Nu> tensor_dom;
    ddc::Chunk tensor_alloc(tensor_dom(), ddc::HostAllocator<double>());
    ddc::ChunkSpan tensor = tensor_alloc.span_view();
    tensor(tensor_dom().front()) = 1;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%f ", tensor(ddc::DiscreteElement<Mu, Nu>(i, j)));
        }
        printf("\n");
    }
    printf("end of test");
}
