// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include "tensor.hpp"

using MetricIndex
    = sil::tensor::LorentzianSignTensorIndex<std::integral_constant<std::size_t, 1>, sil::tensor::MetricIndex1, sil::tensor::MetricIndex2>;

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

using MuUp = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuUp = sil::tensor::TensorContravariantNaturalIndex<Nu>;
using RhoUp = sil::tensor::TensorContravariantNaturalIndex<Rho>;
using SigmaUp = sil::tensor::TensorContravariantNaturalIndex<Sigma>;

using MuLow = sil::tensor::lower<MuUp>;
using NuLow = sil::tensor::lower<NuUp>;
using RhoLow = sil::tensor::lower<RhoUp>;
using SigmaLow = sil::tensor::lower<SigmaUp>;

using RiemannTensorIndex
    = sil::tensor::YoungTableauTensorIndex<
              sil::young_tableau::YoungTableau<
                      4,
                      sil::young_tableau::YoungTableauSeq<
                              std::index_sequence<1, 3>,
                              std::index_sequence<2, 4>>>,
              MuUp,
              NuUp,
              RhoUp,
              SigmaUp>;

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

    ddc::Chunk tensor_low_alloc = ddc::create_mirror_and_copy(tensor);

    auto tensor_low = sil::tensor::inplace_apply_metrics<MetricIndex, ddc::detail::TypeSeq<MuLow, NuLow, RhoLow, SigmaLow>, ddc::detail::TypeSeq<MuUp, NuUp, RhoUp, SigmaUp>>(metric, sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<RiemannTensorIndex>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>(tensor_low_alloc));

    ddc::DiscreteDomain<> dom;
    ddc::Chunk scalar_alloc(dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            scalar(scalar_alloc);

    sil::tensor::tensor_prod2(scalar, tensor_low, tensor);
    std::cout << scalar(ddc::DiscreteElement<>()) << "\n";
}
