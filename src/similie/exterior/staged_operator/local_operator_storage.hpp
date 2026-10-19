// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/concepts.hpp>
#include <similie/exterior/stencil.hpp>
#include <similie/tensor/tensor_prod/tensor_prod.hpp>

namespace sil::exterior {
namespace detail {
// Canonical cochain components are an ordinary vector space: the wedge signs
// and factorial factors are already included in the cached coefficient matrix.
// Retag the existing storage, retaining its stride, rather than expanding forms.
template <class Index, class Tensor>
KOKKOS_FUNCTION auto local_component_tensor(Tensor const& tensor)
{
    static_assert(Tensor::discrete_domain_type::rank() <= 1);
    using Span = ddc::ChunkSpan<
            typename Tensor::element_type,
            ddc::DiscreteDomain<Index>,
            Kokkos::layout_stride,
            typename Tensor::memory_space>;
    std::size_t stride = 1;
    std::size_t front = 0;
    if constexpr (Tensor::discrete_domain_type::rank() == 1) {
        stride = tensor.allocation_mdspan().stride(0);
        front = ddc::detail::array(tensor.domain().front())[0];
    }
    return tensor::Tensor(
            Span(typename Span::allocation_mdspan_type(
                         tensor.data_handle(),
                         typename Span::mapping_type(
                                 typename Span::extents_type(tensor.size()),
                                 std::array<std::size_t, 1> {stride})),
                 ddc::DiscreteDomain<Index>(
                         ddc::DiscreteElement<Index>(front),
                         ddc::DiscreteVector<Index>(tensor.size()))));
}

template <
        class Source,
        class Target,
        template <class...> class Coefficients = tensor::TensorFullIndex,
        class Output,
        class Input,
        class Weights>
KOKKOS_FUNCTION void apply_local_stencil(
        Output const& output,
        Input const& input,
        Weights const& weights,
        std::size_t node)
{
    using SourceComponent = tensor::Covariant<LocalComponentIndex<Source, false>>;
    using TargetComponent = tensor::Covariant<LocalComponentIndex<Target, true>>;
    using MatrixIndex = Coefficients<TargetComponent, tensor::upper_t<SourceComponent>>;
    using MatrixSpan = ddc::ChunkSpan<
            typename Output::element_type const,
            ddc::DiscreteDomain<MatrixIndex>,
            Kokkos::layout_stride,
            typename Output::memory_space>;
    tensor::Tensor matrix(MatrixSpan(
            typename MatrixSpan::allocation_mdspan_type(
                    &weights(node, 0),
                    typename MatrixSpan::mapping_type(
                            typename MatrixSpan::extents_type(MatrixIndex::mem_size()),
                            std::array<std::size_t, 1> {weights.stride(1)})),
            tensor::TensorAccessor<MatrixIndex>::domain()));
    tensor::tensor_prod(
            local_component_tensor<TargetComponent>(output),
            matrix,
            local_component_tensor<SourceComponent>(input));
}

// Apply coefficients computed at the current point through the same diagonal
// tensor product as a cached local operator, without allocating a cache.
template <class Source, class Target, class Output, class Input>
KOKKOS_FUNCTION void apply_local_diagonal(
        Output const& output,
        Input const& input,
        std::array<double, Source::mem_size()> const& weights)
{
    using SourceComponent = tensor::Covariant<LocalComponentIndex<Source, false>>;
    using TargetComponent = tensor::Covariant<LocalComponentIndex<Target, true>>;
    using Matrix = tensor::TensorDiagonalIndex<TargetComponent, tensor::upper_t<SourceComponent>>;
    using Span = ddc::ChunkSpan<
            double const,
            ddc::DiscreteDomain<Matrix>,
            Kokkos::layout_stride,
            typename Output::memory_space>;
    tensor::tensor_prod(
            local_component_tensor<TargetComponent>(output),
            tensor::Tensor(
                    Span(typename Span::allocation_mdspan_type(
                                 weights.data(),
                                 typename Span::mapping_type(
                                         typename Span::extents_type(Matrix::mem_size()),
                                         std::array<std::size_t, 1> {1})),
                         tensor::TensorAccessor<Matrix>::domain())),
            local_component_tensor<SourceComponent>(input));
}

template <class Source, class Target>
KOKKOS_FUNCTION void assert_local_diagonal(
        [[maybe_unused]] LocalOperatorStencil<Source, Target> const& stencil)
{
#ifndef NDEBUG
    double scale = 0.;
    for (double coefficient : stencil.weights)
        scale = Kokkos::max(scale, Kokkos::abs(coefficient));
    for (std::size_t row = 0; row < Target::mem_size(); ++row)
        for (std::size_t col = 0; col < Source::mem_size(); ++col)
            assert((row == col
                    || Kokkos::abs(stencil.weights[row * Source::mem_size() + col])
                               <= 128. * std::numeric_limits<double>::epsilon() * scale)
                   && "Diagonal operator requires diagonal coefficients");
#endif
}


// Bind fixed geometric arguments; the evaluation element remains dynamic.
template <class Operator, class... Geometry>
struct OperatorStencilValue;

template <class Operator>
struct OperatorStencilValue<Operator>
{
    template <class Element, class... Geometry>
    KOKKOS_FUNCTION auto evaluate(Element element, Geometry const&... geometry) const
    {
        return Operator {}.value(geometry..., element);
    }
};

template <class Operator, class First, class... Rest>
struct OperatorStencilValue<Operator, First, Rest...>
{
    First first;
    OperatorStencilValue<Operator, Rest...> rest;

    KOKKOS_FUNCTION OperatorStencilValue(First const& first_, Rest const&... rest_)
        : first(first_)
        , rest(rest_...)
    {
    }

    template <class Element, class... Bound>
    KOKKOS_FUNCTION auto evaluate(Element element, Bound const&... bound) const
    {
        return rest.evaluate(element, bound..., first);
    }

    template <class Element>
    KOKKOS_FUNCTION auto operator()(Element element) const
    {
        return evaluate(element);
    }
};

// The policy retains the native coefficient index: dense and diagonal maps do
// not acquire a common expanded representation during either build or apply.
template <
        class Operator,
        class Source,
        class Target,
        class Domain,
        class Memory,
        class Exec = typename Memory::execution_space,
        template <class...> class Coefficients = tensor::TensorFullIndex,
        class SourceElement = typename tensor::natural_domain_t<Source>::discrete_element_type,
        class TargetElement = typename tensor::natural_domain_t<Target>::discrete_element_type,
        bool NormalizeSource = true>
class LocalOperatorStorage
{
    using Element = typename Domain::discrete_element_type;
    LocalStencilStorage<Source, Target, Domain, Memory, Coefficients> m_stencils;

    template <class... Geometry>
    static LocalOperatorStorage cache(
            Exec const& exec,
            Domain const& domain,
            Geometry const&... geometry)
    {
        LocalOperatorStorage result;
        result.m_stencils = cache_local_stencils<Source, Target, Memory, Coefficients>(
                exec,
                domain,
                OperatorStencilValue<Operator, Geometry...>(geometry...));
        return result;
    }

public:
    using execution_space = Exec;
    template <class OtherExec, template <class...> class OtherCoefficients = Coefficients>
    using rebind = LocalOperatorStorage<
            Operator,
            Source,
            Target,
            Domain,
            Memory,
            OtherExec,
            OtherCoefficients,
            SourceElement,
            TargetElement,
            NormalizeSource>;

    template <class Geometry>
        requires requires(Geometry const& geometry, Element element) {
            Operator {}.value(geometry, element);
        }
    static LocalOperatorStorage build(Exec const& exec, Geometry const& geometry)
    {
        return cache(exec, geometry.non_indices_domain(), geometry);
    }

    template <class Geometry1, class Geometry2>
        requires requires(Geometry1 const& first, Geometry2 const& second, Element element) {
            Operator {}.value(first, second, element);
        }
    static LocalOperatorStorage build(
            Exec const& exec,
            Geometry1 const& first,
            Geometry2 const& second)
    {
        return cache(exec, first.non_indices_domain(), first, second);
    }

    template <class Geometry>
        requires requires(Geometry const& geometry, Element element) {
            Operator {}.value(geometry, element);
        }
    static LocalOperatorStorage build(
            Exec const& exec,
            Geometry const& geometry,
            Domain const& domain)
    {
        return cache(exec, domain, geometry);
    }

    template <class Geometry1, class Geometry2>
        requires requires(Geometry1 const& first, Geometry2 const& second, Element element) {
            Operator {}.value(first, second, element);
        }
    static LocalOperatorStorage build(
            Exec const& exec,
            Geometry1 const& first,
            Geometry2 const& second,
            Domain const& domain)
    {
        return cache(exec, domain, first, second);
    }

    template <
            misc::Specialization<tensor::Tensor> Output,
            misc::Specialization<tensor::Tensor> Input,
            class... Geometry>
        requires(
                sizeof...(Geometry) > 0
                && requires(
                        Geometry const&... geometry,
                        Element element) { Operator {}.value(geometry..., element); })
    static LocalOperatorStorage build(
            Exec const& exec,
            Output const& output,
            Input const&,
            Geometry const&... geometry)
    {
        return cache(exec, output.non_indices_domain(), geometry...);
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void apply_pointwise(Output const& output, Input const& input, Element element)
            const
    {
        apply_local_stencil<Source, Target, Coefficients>(
                output,
                input,
                m_stencils.weights,
                stencil_node(m_stencils.domain, element));
    }

    template <class Output, class Input, class Geometry>
    KOKKOS_FUNCTION void apply_pointwise(
            Output const& output,
            Input const& input,
            Geometry const&,
            Element element) const
    {
        apply_pointwise(output, input, element);
    }

    template <class Output, class Input, class Geometry1, class Geometry2>
    KOKKOS_FUNCTION void apply_pointwise(
            Output const& output,
            Input const& input,
            Geometry1 const&,
            Geometry2 const&,
            Element element) const
    {
        apply_pointwise(output, input, element);
    }

    template <
            misc::Specialization<tensor::Tensor> Output,
            misc::Specialization<tensor::Tensor> Input>
    Output apply(Exec const& exec, Output const& output, Input const& input) const
    {
        LocalOperatorStorage const storage = *this;
        ddc::parallel_for_each(
                "similie_apply_staged_local_operator",
                exec,
                output.non_indices_domain(),
                KOKKOS_LAMBDA(Element element) {
                    storage.apply_pointwise(output[element], input[element], element);
                });
        return output;
    }

    KOKKOS_FUNCTION LocalOperatorStencil<Source, Target> value(Element element) const
    {
        LocalOperatorStencil<Source, Target> result;
        using Index = LocalCoefficientIndex<Source, Target, Coefficients>;
        std::size_t const node = stencil_node(m_stencils.domain, element);
        for (std::size_t entry = 0; entry < Index::mem_size(); ++entry) {
            std::array const ids = Index::mem_id_to_canonical_natural_ids(entry);
            result.weights[ids[0] * Source::mem_size() + ids[1]] = m_stencils.weights(node, entry);
        }
        return result;
    }

    template <class NaturalElement>
    KOKKOS_FUNCTION double value(Element element, NaturalElement natural) const
    {
        return local_stencil_coefficient<Source, Target, NormalizeSource>(
                value(element),
                ddc::detail::array(SourceElement(natural)),
                ddc::detail::array(TargetElement(natural)));
    }

    template <class Geometry>
    KOKKOS_FUNCTION LocalOperatorStencil<Source, Target> value(Geometry const&, Element element)
            const
    {
        return value(element);
    }

    template <class Geometry, class NaturalElement>
    KOKKOS_FUNCTION double value(Geometry const&, Element element, NaturalElement natural) const
    {
        return value(element, natural);
    }

    template <class Geometry1, class Geometry2>
    KOKKOS_FUNCTION LocalOperatorStencil<Source, Target> value(
            Geometry1 const&,
            Geometry2 const&,
            Element element) const
    {
        return value(element);
    }

    template <class Geometry1, class Geometry2, class NaturalElement>
    KOKKOS_FUNCTION double value(
            Geometry1 const&,
            Geometry2 const&,
            Element element,
            NaturalElement natural) const
    {
        return value(element, natural);
    }
};

// Use only when the operator type proves that its coefficients do not depend
// on geometry. The retained descriptor supplies domain/type information, not
// a per-node coefficient allocation.
template <
        class Operator,
        class Geometry,
        class Exec = typename Geometry::memory_space::execution_space>
class ImplicitLocalOperatorStorage
{
    using Element = typename Geometry::non_indices_domain_t::discrete_element_type;
    Geometry m_geometry;

    explicit ImplicitLocalOperatorStorage(Geometry const& geometry) : m_geometry(geometry) {}

public:
    using execution_space = Exec;
    template <class OtherExec>
    using rebind = ImplicitLocalOperatorStorage<Operator, Geometry, OtherExec>;

    static ImplicitLocalOperatorStorage build(Exec const&, Geometry const& geometry)
    {
        return ImplicitLocalOperatorStorage(geometry);
    }

    template <class Output, class Input>
    static ImplicitLocalOperatorStorage build(
            Exec const& exec,
            Output const&,
            Input const&,
            Geometry const& geometry)
    {
        return build(exec, geometry);
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void apply_pointwise(Output const& output, Input const& input, Element element)
            const
    {
        Operator {}(output, input, m_geometry, element);
    }

    template <class Output, class Input>
    KOKKOS_FUNCTION void apply_pointwise(
            Output const& output,
            Input const& input,
            Geometry const&,
            Element element) const
    {
        apply_pointwise(output, input, element);
    }

    template <
            misc::Specialization<tensor::Tensor> Output,
            misc::Specialization<tensor::Tensor> Input>
    Output apply(Exec const& exec, Output const& output, Input const& input) const
    {
        ImplicitLocalOperatorStorage const storage = *this;
        ddc::parallel_for_each(
                "similie_apply_staged_implicit_operator",
                exec,
                output.non_indices_domain(),
                KOKKOS_LAMBDA(Element element) {
                    storage.apply_pointwise(output[element], input[element], element);
                });
        return output;
    }

    KOKKOS_FUNCTION auto value(Geometry, Element element) const
    {
        return Operator {}.value(m_geometry, element);
    }

    template <class NaturalElement>
    KOKKOS_FUNCTION double value(Geometry, Element element, NaturalElement natural) const
    {
        return Operator {}.value(m_geometry, element, natural);
    }
};

} // namespace detail
} // namespace sil::exterior
