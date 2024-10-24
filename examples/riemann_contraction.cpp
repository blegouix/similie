// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include "tensor.hpp"

struct T
{
};

struct X
{
};

struct Y
{
};

struct Z
{
};

struct Mu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Nu : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Rho : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct Sigma : sil::tensor::TensorNaturalIndex<T, X, Y, Z>
{
};

struct RiemannTensorIndex
    : sil::tensor::YoungTableauTensorIndex<
              sil::young_tableau::YoungTableau<
                      4,
                      sil::young_tableau::YoungTableauSeq<
                              std::index_sequence<1, 3>,
                              std::index_sequence<2, 4>>>,
              Mu,
              Nu,
              Rho,
              Sigma>
{
};

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    printf("start example\n");

    ddc::DiscreteDomain<RiemannTensorIndex> tensor_dom(
            ddc::DiscreteElement<RiemannTensorIndex>(0),
            ddc::DiscreteVector<RiemannTensorIndex>(20));
    ddc::Chunk tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<RiemannTensorIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tensor(tensor_alloc);

    for (std::size_t i = 0; i < 20; ++i) {
        tensor.mem(ddc::DiscreteElement<RiemannTensorIndex>(i)) = 1.;
    }

    ddc::DiscreteDomain<> dom;
    ddc::Chunk scalar_alloc(dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            scalar(scalar_alloc);

    sil::tensor::tensor_prod2(scalar, tensor, tensor);
}
