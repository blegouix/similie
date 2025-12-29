// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

/**
 * @brief Helper functor to bridge a C++ DiscreteElement with an external callable.
 *
 * The evaluator stores a callable (for example, a function pointer produced by a Cython
 * module) and exposes an operator() that forwards the uids of a DiscreteElement for the
 * ordered indices of a discrete domain to that callable.
 *
 * @tparam Functor Callable type invoked with the uids of the discrete domain @p Dom.
 * @tparam Dom Discrete domain whose `discrete_element_type` defines the element forwarded.
 */
template <class Functor, class Dom>
class FunctorEvaluatorByOrderedIndices
{
public:
    using functor_type = Functor;
    using element_type = typename Dom::discrete_element_type;

    KOKKOS_FUNCTION explicit FunctorEvaluatorByOrderedIndices(Functor functor)
        : m_functor(std::move(functor))
    {
    }

    KOKKOS_FUNCTION FunctorEvaluatorByOrderedIndices(FunctorEvaluatorByOrderedIndices const&)
            = default;
    KOKKOS_FUNCTION FunctorEvaluatorByOrderedIndices(FunctorEvaluatorByOrderedIndices&&)
            = default;
    KOKKOS_FUNCTION FunctorEvaluatorByOrderedIndices&
    operator=(FunctorEvaluatorByOrderedIndices const&)
            = default;
    KOKKOS_FUNCTION FunctorEvaluatorByOrderedIndices&
    operator=(FunctorEvaluatorByOrderedIndices&&)
            = default;
    ~FunctorEvaluatorByOrderedIndices() = default;

    /**
     * @brief Invoke the underlying callable with the uids of a DiscreteElement.
     *
     * @param elem The discrete element containing the indices to forward.
     * @return The result of the underlying callable.
     */
    KOKKOS_FUNCTION auto operator()(element_type const& elem) const
            -> decltype(invoke(elem, ddc::to_type_seq_t<Dom>()))
    {
        return invoke(elem, ddc::to_type_seq_t<Dom>());
    }

private:
    template <class Elem, class... DDim>
    KOKKOS_FUNCTION auto invoke(Elem const& elem, ddc::detail::TypeSeq<DDim...>) const
            -> decltype(std::declval<Functor const&>()(elem.template uid<DDim>()...))
    {
        return m_functor(elem.template uid<DDim>()...);
    }

    Functor m_functor;
};

} // namespace misc

} // namespace sil
