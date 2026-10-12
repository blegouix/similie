// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <utility>
#include <vector>

#include "multidomain.hpp"

namespace sil::multidomains {
/** Dispatch any homogeneous collection of domain operators.
 * \important This operator and documentation is fully AI-generated.
 * Each entry retains its ordinary operator/value API. Batch application gives
 * each domain a partitioned execution space and fences before releasing those
 * spaces. value(id, ...) forwards to the selected operator; operator[](id)
 * provides a copyable local operator for pointwise evaluation in kernels.
 */
template <class Operator>
class MultidomainOperator
{
    std::vector<Operator> m_operators;

public:
    explicit MultidomainOperator(std::vector<Operator> operators)
        : m_operators(std::move(operators))
    {
    }
    Operator const& operator[](std::size_t id) const
    {
        return m_operators.at(id);
    }
    std::size_t domain_count() const
    {
        return m_operators.size();
    }
    template <class... Arguments>
    decltype(auto) value(std::size_t id, Arguments&&... arguments) const
    {
        return m_operators.at(id).value(std::forward<Arguments>(arguments)...);
    }
    template <class... Arguments>
    decltype(auto) operator()(std::size_t id, Arguments&&... arguments) const
    {
        return m_operators.at(id)(std::forward<Arguments>(arguments)...);
    }
    template <class ExecSpace, class Function>
    void for_each(ExecSpace const& exec, Function function) const
    {
        DomainExecution<ExecSpace> partitions(exec, m_operators.size());
        partitions.for_each([&](std::size_t id, ExecSpace const& space) {
            function(id, space, m_operators[id]);
        });
        partitions.fence();
    }
    template <class ExecSpace, class Output, class Input, class... Arguments>
        requires(Kokkos::is_execution_space<ExecSpace>::value)
    void operator()(
            ExecSpace const& exec,
            Output const& output,
            Input const& input,
            Arguments const&... arguments) const
    {
        for_each(exec, [&](std::size_t id, ExecSpace const& space, Operator const& operation) {
            operation(space, output.at(id), input.at(id), arguments...);
        });
    }
};
template <class Operator>
MultidomainOperator(std::vector<Operator>) -> MultidomainOperator<Operator>;

/** Replicate a local differential operator for each physical domain.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Domains, class Operator>
MultidomainOperator<Operator> make_multidomain_operator(Domains const& domains, Operator operation)
{
    return MultidomainOperator<Operator>(std::vector<Operator>(domains.domain_count(), operation));
}
} // namespace sil::multidomains
