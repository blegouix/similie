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

    ddc::DiscreteDomain<MuUp, NuUp, RhoUp, SigmaUp> natural_tensor_dom(
            ddc::DiscreteElement<MuUp, NuUp, RhoUp, SigmaUp>(0, 0, 0, 0),
            ddc::DiscreteVector<MuUp, NuUp, RhoUp, SigmaUp>(4, 4, 4, 4));
    ddc::Chunk natural_tensor_alloc(natural_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<MuUp, NuUp, RhoUp, SigmaUp>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            natural_tensor(natural_tensor_alloc);
    ddc::parallel_fill(natural_tensor, 0.);
    natural_tensor(natural_tensor.accessor().element<T,X,T,X>()) = -.25;
    natural_tensor(natural_tensor.accessor().element<T,X,X,T>()) = .25;
    natural_tensor(natural_tensor.accessor().element<T,Y,T,Y>()) = -1.;
    natural_tensor(natural_tensor.accessor().element<T,Y,X,Y>()) = .5;
    natural_tensor(natural_tensor.accessor().element<T,Y,Y,T>()) = 1.;
    natural_tensor(natural_tensor.accessor().element<T,Y,Y,X>()) = -.5;
    natural_tensor(natural_tensor.accessor().element<T,Z,T,Z>()) = -1.;
    natural_tensor(natural_tensor.accessor().element<T,Z,X,Z>()) = .5;
    natural_tensor(natural_tensor.accessor().element<T,Z,Z,T>()) = 1.;
    natural_tensor(natural_tensor.accessor().element<T,Z,Z,X>()) = -.5;

    natural_tensor(natural_tensor.accessor().element<X,T,T,X>()) = .25;
    natural_tensor(natural_tensor.accessor().element<X,T,X,T>()) = -.25;
    natural_tensor(natural_tensor.accessor().element<X,Y,T,Y>()) = -1.5;
    natural_tensor(natural_tensor.accessor().element<X,Y,X,Y>()) = 1.;
    natural_tensor(natural_tensor.accessor().element<X,Y,Y,T>()) = 1.5;
    natural_tensor(natural_tensor.accessor().element<X,Y,Y,X>()) = -1.;
    natural_tensor(natural_tensor.accessor().element<X,Z,T,Z>()) = -1.5;
    natural_tensor(natural_tensor.accessor().element<X,Z,X,Z>()) = 1.;
    natural_tensor(natural_tensor.accessor().element<X,Z,Z,T>()) = 1.5;
    natural_tensor(natural_tensor.accessor().element<X,Z,Z,X>()) = -1.;

    natural_tensor(natural_tensor.accessor().element<Y,T,T,Y>()) = 1.;
    natural_tensor(natural_tensor.accessor().element<Y,T,X,Y>()) = -1.5;
    natural_tensor(natural_tensor.accessor().element<Y,T,Y,T>()) = -1.0;
    natural_tensor(natural_tensor.accessor().element<Y,T,Y,X>()) = 1.5;
    natural_tensor(natural_tensor.accessor().element<Y,X,T,Y>()) = -1.5;
    natural_tensor(natural_tensor.accessor().element<Y,X,X,Y>()) = 2.;
    natural_tensor(natural_tensor.accessor().element<Y,X,Y,T>()) = 1.5;
    natural_tensor(natural_tensor.accessor().element<Y,X,Y,X>()) = -2.;

    natural_tensor(natural_tensor.accessor().element<Z,T,T,Z>()) = 1.;
    natural_tensor(natural_tensor.accessor().element<Z,T,X,Z>()) = -1.5;
    natural_tensor(natural_tensor.accessor().element<Z,T,Z,T>()) = -1.0;
    natural_tensor(natural_tensor.accessor().element<Z,T,Z,X>()) = 1.5;
    natural_tensor(natural_tensor.accessor().element<Z,X,T,Z>()) = -1.5;
    natural_tensor(natural_tensor.accessor().element<Z,X,X,Z>()) = 2.;
    natural_tensor(natural_tensor.accessor().element<Z,X,Z,T>()) = 1.5;
    natural_tensor(natural_tensor.accessor().element<Z,X,Z,X>()) = -2.;
    std::cout << natural_tensor;

    sil::tensor::compress(tensor, natural_tensor);
/*
[[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 [[0, -0.25, 0, 0], [0.25, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
[[0, 0, -1.0, 0], [0, 0, 0.5, 0], [1.0, -0.5, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, -1.0], [0, 0, 0, 0.5], [0, 0, 0, 1.22464679914735e-16], [1.0, -0.5, -1.22464679914735e-16, 0]]],

 [[[0, 0.25, 0, 0], [-0.25, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 [[0, 0, -1.5, 0], [0, 0, 1.0, 0], [1.5, -1.0, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, -1.5], [0, 0, 0, 1.0], [0, 0, 0, 1.22464679914735e-16], [1.5, -1.0, -1.22464679914735e-16, 0]]],

 [[[0, 0, 1.0, 0], [0, 0, -1.5, 0], [-1.0, 1.5, 0, 0], [0, 0, 0, 0]],
 [[0, 0, -1.5, 0], [0, 0, 2.0, 0], [1.5, -2.0, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3.74939945665464e-33], [0, 0, -3.74939945665464e-33, 0]]],

 [[[0, 0, 0, 1.0], [0, 0, 0, -1.5], [0, 0, 0, 0], [-1.0, 1.5, 0, 0]],
 [[0, 0, 0, -1.5], [0, 0, 0, 2.0], [0, 0, 0, 0], [1.5, -2.0, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3.74939945665464e-33], [0, 0, -3.74939945665464e-33, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]]
*/

/*
    for (std::size_t i = 0; i < 20; ++i) {
        tensor.mem(ddc::DiscreteElement<RiemannTensorIndex>(i)) = 1.;
    }
*/
    std::cout << tensor;
    for (std::size_t i = 0; i < 20; ++i) {
        std::cout << tensor.mem(ddc::DiscreteElement<RiemannTensorIndex>(i)) << "\n";
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
