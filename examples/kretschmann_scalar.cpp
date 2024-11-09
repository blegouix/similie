// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#include <ddc/ddc.hpp>

#include "tensor.hpp"

/**
 * This example computes the Kretschmann scalar R_mu_nu_rho_sigma*R^mu^nu^rho^sigma on the horizon of a black hole, which is known to be 12.
 * The Riemann tensor R is pre-computed using the python script https://github.com/blegouix/python-black-hole/blob/main/lemaitre_horizon.py in the Lemaitre coordinate system, which is non-singular on the horizon.
 * On the particular event {tau: 1/3, rho: 1, theta: math.pi/2, phi: 0}, the metric is Minkowski metric.
 */

// Declare the Minkowski metric as a Lorentzian signature (-, +, +, +)
using MetricIndex = sil::tensor::LorentzianSignTensorIndex<
        std::integral_constant<std::size_t, 1>,
        sil::tensor::MetricIndex1,
        sil::tensor::MetricIndex2>;

// Labelize the dimensions of space-time
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

// Declare natural indices taking values in {T, X, Y, Z}
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

// Declare upper (contravariant) indices
using MuUp = sil::tensor::TensorContravariantNaturalIndex<Mu>;
using NuUp = sil::tensor::TensorContravariantNaturalIndex<Nu>;
using RhoUp = sil::tensor::TensorContravariantNaturalIndex<Rho>;
using SigmaUp = sil::tensor::TensorContravariantNaturalIndex<Sigma>;

// Declare also their covariant counterparts
using MuLow = sil::tensor::lower<MuUp>;
using NuLow = sil::tensor::lower<NuUp>;
using RhoLow = sil::tensor::lower<RhoUp>;
using SigmaLow = sil::tensor::lower<SigmaUp>;

// Declare a unique index for fully-contravariant Riemann tensor, satisfying Riemann symmetries (cf. https://birdtracks.eu/ section 10.5)
using RiemannUpTensorIndex = sil::tensor::YoungTableauTensorIndex<
        sil::young_tableau::YoungTableau<
                4,
                sil::young_tableau::
                        YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2, 4>>>,
        MuUp,
        NuUp,
        RhoUp,
        SigmaUp>;

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    // Allocate and instantiate a metric tensor. Because the metric in Lorentzian, the size of the allocation is 0 (metric_dom is empty).
    sil::tensor::TensorAccessor<MetricIndex> metric_accessor;
    ddc::DiscreteDomain<MetricIndex> metric_dom = metric_accessor.mem_domain();
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<MetricIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            metric(metric_alloc);

    // Allocate and instantiate a fully-contravariant Riemann tensor. The size of the allocation is 20 because the Riemann tensor has 20 independant components.
    sil::tensor::TensorAccessor<RiemannUpTensorIndex> riemann_up_accessor;
    ddc::DiscreteDomain<RiemannUpTensorIndex> riemann_up_dom = riemann_up_accessor.mem_domain();
    ddc::Chunk riemann_up_alloc(riemann_up_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<RiemannUpTensorIndex>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            riemann_up(riemann_up_alloc);

    // Young-tableau-indexed tensors cannot be filled directly (because the 20 independant components do not appear explicitely in the 4^4 Riemann tensor. The explicit components of the Riemann tensor are linear combinations of the 20 independant components). We thus need to allocate a naturally-indexed 4^4 tensor to be filled.
    ddc::DiscreteDomain<MuUp, NuUp, RhoUp, SigmaUp> natural_tensor_dom(
            ddc::DiscreteElement<MuUp, NuUp, RhoUp, SigmaUp>(0, 0, 0, 0),
            ddc::DiscreteVector<MuUp, NuUp, RhoUp, SigmaUp>(4, 4, 4, 4));
    ddc::Chunk natural_tensor_alloc(natural_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<MuUp, NuUp, RhoUp, SigmaUp>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            natural_tensor(natural_tensor_alloc);
    ddc::parallel_fill(natural_tensor, 0.);

    // We fill the naturally-indexed tensor with the pre-computed values of the explicit Riemann tensor components on the horizon in the Lemaitre coordinate system (cf. the python script)
    natural_tensor(natural_tensor.access_element<T, X, T, X>()) = -1.;
    natural_tensor(natural_tensor.access_element<T, X, X, T>()) = 1.;
    natural_tensor(natural_tensor.access_element<T, Y, T, Y>()) = .5;
    natural_tensor(natural_tensor.access_element<T, Y, Y, T>()) = -.5;
    natural_tensor(natural_tensor.access_element<T, Z, T, Z>()) = .5;
    natural_tensor(natural_tensor.access_element<T, Z, Z, T>()) = -.5;

    natural_tensor(natural_tensor.access_element<X, T, T, X>()) = 1.;
    natural_tensor(natural_tensor.access_element<X, T, X, T>()) = -1.;
    natural_tensor(natural_tensor.access_element<X, Y, X, Y>()) = -.5;
    natural_tensor(natural_tensor.access_element<X, Y, Y, X>()) = .5;
    natural_tensor(natural_tensor.access_element<X, Z, X, Z>()) = -.5;
    natural_tensor(natural_tensor.access_element<X, Z, Z, X>()) = .5;

    natural_tensor(natural_tensor.access_element<Y, T, T, Y>()) = -.5;
    natural_tensor(natural_tensor.access_element<Y, T, Y, T>()) = .5;
    natural_tensor(natural_tensor.access_element<Y, X, X, Y>()) = .5;
    natural_tensor(natural_tensor.access_element<Y, X, Y, X>()) = -.5;
    natural_tensor(natural_tensor.access_element<Y, Z, Y, Z>()) = 1.;
    natural_tensor(natural_tensor.access_element<Y, Z, Z, Y>()) = -1.;

    natural_tensor(natural_tensor.access_element<Z, T, T, Z>()) = -.5;
    natural_tensor(natural_tensor.access_element<Z, T, Z, T>()) = .5;
    natural_tensor(natural_tensor.access_element<Z, X, X, Z>()) = .5;
    natural_tensor(natural_tensor.access_element<Z, X, Z, X>()) = -.5;
    natural_tensor(natural_tensor.access_element<Z, Y, Y, Z>()) = -1.;
    natural_tensor(natural_tensor.access_element<Z, Y, Z, Y>()) = 1.;

    // We "compress" the 256 components of the naturally-indexed tensor into the 20 independent components of the Young-tableau-indexed Riemann tensor.
    sil::tensor::compress(riemann_up, natural_tensor);

    /*
    for (std::size_t i = 0; i < 20; ++i) {
        std::cout << riemann_up.mem(ddc::DiscreteElement<RiemannTensorIndex>(i)) << "\n";
    }
    */

    // We allocate and compute the covariant counterpart of the Riemann tensor which is needed for the computation of the Kretschmann scalar. Actually, in this particular case (Minkowski metric and even-rank tensor), contravariant and covariant Riemann tensors have the same components, but we perform the computation like if it was not the case.
    ddc::Chunk riemann_low_alloc = ddc::create_mirror_and_copy(riemann_up);
    auto riemann_low = sil::tensor::inplace_apply_metrics<
            MetricIndex,
            ddc::detail::TypeSeq<MuLow, NuLow, RhoLow, SigmaLow>,
            ddc::detail::TypeSeq<MuUp, NuUp, RhoUp, SigmaUp>>(
            sil::tensor::Tensor<
                    double,
                    ddc::DiscreteDomain<RiemannUpTensorIndex>,
                    Kokkos::layout_right,
                    Kokkos::DefaultHostExecutionSpace::memory_space>(riemann_low_alloc),
            metric);

    // We allocate the Kretschmann scalar (a single double) and perform the tensor product between covariant an contravariant Riemann tensors.
    ddc::DiscreteDomain<> dom;
    ddc::Chunk scalar_alloc(dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            scalar(scalar_alloc);
    sil::tensor::tensor_prod2(scalar, riemann_low, riemann_up);
    std::cout << "Kreschmann scalar = " << scalar(ddc::DiscreteElement<>()) << "\n";
}
