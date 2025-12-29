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
 * The Evaluator stores a callable (for example, a function pointer produced by a Cython
 * module) and exposes an operator() that forwards the uids of a DiscreteElement for the
 * requested dimensions to that callable.
 *
 * @tparam Functor Callable type invoked with the uids of @p DDim.
 * @tparam DDim Parameter pack of discrete dimensions or tensor indices.
 */
template <class Functor, class... DDim>
class DiscreteElementEvaluator
{
public:
    using functor_type = Functor;

    KOKKOS_FUNCTION explicit DiscreteElementEvaluator(Functor functor)
        : m_functor(std::move(functor))
    {
    }

    KOKKOS_FUNCTION DiscreteElementEvaluator(DiscreteElementEvaluator const&) = default;
    KOKKOS_FUNCTION DiscreteElementEvaluator(DiscreteElementEvaluator&&) = default;
    KOKKOS_FUNCTION DiscreteElementEvaluator&
    operator=(DiscreteElementEvaluator const&)
            = default;
    KOKKOS_FUNCTION DiscreteElementEvaluator& operator=(DiscreteElementEvaluator&&) = default;
    ~DiscreteElementEvaluator() = default;

    /**
     * @brief Invoke the underlying callable with the uids of a DiscreteElement.
     *
     * @param elem The discrete element containing the indices to forward.
     * @return The result of the underlying callable.
     */
    template <class Elem>
    KOKKOS_FUNCTION auto operator()(Elem const& elem) const
            -> decltype(std::declval<Functor const&>()(elem.template uid<DDim>()...))
    {
        return m_functor(elem.template uid<DDim>()...);
    }

private:
    Functor m_functor;
};

template <class... DDim, class Functor>
KOKKOS_FUNCTION auto make_discrete_element_evaluator(Functor&& functor)
{
    return DiscreteElementEvaluator<std::decay_t<Functor>, DDim...>(
            std::forward<Functor>(functor));
}

} // namespace misc

} // namespace sil
