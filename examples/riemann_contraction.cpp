// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include "tensor.hpp"

//struct MetricIndex : sil::tensor::MetricTensorIndex<sil::tensor::LorentzianSignTensorIndex, sil::tensor::detail::DummyIndex1, sil::tensor::detail::DummyIndex1, 2>
struct MetricIndex
    : sil::tensor::IdentityTensorIndex<sil::tensor::MetricIndex1, sil::tensor::MetricIndex2>
{
};

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

struct Mu : sil::tensor::TensorContravariantNaturalIndex<T, X, Y, Z>
{
};

struct Nu : sil::tensor::TensorContravariantNaturalIndex<T, X, Y, Z>
{
};

struct Rho : sil::tensor::TensorContravariantNaturalIndex<T, X, Y, Z>
{
};

struct Sigma : sil::tensor::TensorContravariantNaturalIndex<T, X, Y, Z>
{
};

using MuLow = typename sil::tensor::Lower<Mu>;

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

    sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<MetricIndex> metric_dom = metric_accessor.mem_domain();
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<MetricIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric(metric_alloc);

    //sil::tensor::Tensor metric_mu = sil::tensor::relabelize_metric<MetricIndex, MuLow, Mu>(metric);
    // std::cout << metric_mu;

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

    sil::tensor::inplace_apply_metric<MetricIndex, MuLow, Mu>(metric, tensor);

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
