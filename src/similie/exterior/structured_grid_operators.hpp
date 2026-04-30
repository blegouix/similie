// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/domain_contains.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/tensor/identity_tensor.hpp>
#include <similie/tensor/tensor_impl.hpp>

namespace sil::exterior {

namespace detail {

template <class DDim, class Domain>
KOKKOS_FUNCTION double uniform_cell_length(Domain const& domain)
{
    auto const subdomain = ddc::DiscreteDomain<DDim>(domain);
    if (subdomain.template extent<DDim>() <= 1) {
        return 1.;
    }
    auto const first = subdomain.front();
    auto const second = first + ddc::DiscreteVector<DDim>(1);
    return static_cast<double>(ddc::coordinate(second) - ddc::coordinate(first));
}

template <std::size_t I, class DDimSeq>
struct StructuredGradient;

template <std::size_t I>
struct StructuredGradient<I, ddc::detail::TypeSeq<>>
{
    template <class ExecSpace, class OutTensor, class InTensor>
    static void run(ExecSpace const&, OutTensor, InTensor)
    {
    }
};

template <std::size_t I, class DDim, class... Tail>
struct StructuredGradient<I, ddc::detail::TypeSeq<DDim, Tail...>>
{
    template <class ExecSpace, class OutTensor, class InTensor>
    static void run(ExecSpace const& exec_space, OutTensor out, InTensor in)
    {
        double const dx = uniform_cell_length<DDim>(in.non_indices_domain());
        auto const in_component = in.indices_domain().front();
        auto const out_component = typename OutTensor::indices_domain_t::discrete_element_type(I);

        ddc::parallel_for_each(
                "similie_structured_gradient_component",
                exec_space,
                out.non_indices_domain(),
                KOKKOS_LAMBDA(
                        typename OutTensor::non_indices_domain_t::discrete_element_type elem) {
                    auto const next = elem + ddc::DiscreteVector<DDim>(1);
                    double value = 0.;
                    if (misc::domain_contains(in.non_indices_domain(), next)) {
                        value = (in.mem(next, in_component) - in.mem(elem, in_component)) / dx;
                    }
                    out.mem(elem, out_component) = value;
                });

        StructuredGradient<I + 1, ddc::detail::TypeSeq<Tail...>>::run(exec_space, out, in);
    }
};

template <std::size_t I, class DDimSeq>
struct StructuredDivergence;

template <std::size_t I>
struct StructuredDivergence<I, ddc::detail::TypeSeq<>>
{
    template <class InTensor, class Elem>
    KOKKOS_FUNCTION static double run(InTensor, Elem)
    {
        return 0.;
    }
};

template <std::size_t I, class DDim, class... Tail>
struct StructuredDivergence<I, ddc::detail::TypeSeq<DDim, Tail...>>
{
    template <class InTensor, class Elem>
    KOKKOS_FUNCTION static double run(InTensor in, Elem elem)
    {
        auto const previous = elem - ddc::DiscreteVector<DDim>(1);
        auto const component = typename InTensor::indices_domain_t::discrete_element_type(I);
        double const dx = uniform_cell_length<DDim>(in.non_indices_domain());
        double const current_flux = in.mem(elem, component);
        double const previous_flux = misc::domain_contains(in.non_indices_domain(), previous)
                                             ? in.mem(previous, component)
                                             : 0.;

        return (current_flux - previous_flux) / dx
               + StructuredDivergence<I + 1, ddc::detail::TypeSeq<Tail...>>::run(in, elem);
    }
};

} // namespace detail

template <
        tensor::TensorNatIndex GradientIndex,
        tensor::TensorIndex ScalarIndex,
        misc::Specialization<tensor::Tensor> OutTensor,
        misc::Specialization<tensor::Tensor> InTensor,
        class ExecSpace>
OutTensor structured_coefficient_gradient(ExecSpace const& exec_space, OutTensor out, InTensor in)
{
    static_assert(GradientIndex::rank() == 1);
    static_assert(ScalarIndex::rank() == 0);

    detail::StructuredGradient<0, ddc::to_type_seq_t<typename InTensor::non_indices_domain_t>>::
            run(exec_space, out, in);
    return out;
}

template <
        tensor::TensorIndex MetricIndex,
        tensor::TensorNatIndex OneFormIndex,
        misc::Specialization<tensor::Tensor> OutTensor,
        misc::Specialization<tensor::Tensor> InTensor,
        misc::Specialization<tensor::Tensor> MetricTensor,
        class ExecSpace>
OutTensor structured_identity_metric_divergence(
        ExecSpace const& exec_space,
        OutTensor out,
        InTensor in,
        MetricTensor const&)
{
    static_assert(OneFormIndex::rank() == 1);
    static_assert(std::is_same_v<
                  typename OutTensor::indices_domain_t::discrete_element_type,
                  ddc::DiscreteElement<tensor::Covariant<tensor::ScalarIndex>>>);
    static_assert(misc::Specialization<MetricIndex, tensor::TensorIdentityIndex>);

    auto const scalar_component = out.indices_domain().front();
    ddc::parallel_for_each(
            "similie_structured_identity_metric_divergence",
            exec_space,
            out.non_indices_domain(),
            KOKKOS_LAMBDA(typename OutTensor::non_indices_domain_t::discrete_element_type elem) {
                out.mem(elem, scalar_component) = detail::StructuredDivergence<
                        0,
                        ddc::to_type_seq_t<typename InTensor::non_indices_domain_t>>::run(in, elem);
            });

    return out;
}

} // namespace sil::exterior
