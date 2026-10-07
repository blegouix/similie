// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

namespace sil::multidomains {

/** One execution-space instance per physical domain; boundary nodes launch no work.
 * \important This operator and documentation is fully AI-generated.
 * On CUDA partition_space creates one owned stream per domain. for_each() only
 * submits work; fence() joins all domains at an explicit phase boundary. Inputs
 * must be ready before submission, and donor writes must finish before reads
 * from another domain. Keep this object alive until its submitted work completes.
 */
template <class Graph, class ExecSpace>
class DomainExecution
{
    std::vector<ExecSpace> m_spaces;

public:
    explicit DomainExecution(ExecSpace const& parent)
    {
        std::size_t count = 0;
        Graph::for_each_domain([&]<class Node>() { ++count; });
        parent.fence("multidomain input readiness");
        m_spaces = Kokkos::Experimental::partition_space(parent, std::vector<int>(count, 1));
    }

    template <class Function>
    void for_each(Function function) const
    {
        std::size_t index = 0;
        Graph::for_each_domain(
                [&]<class Node>() { function.template operator()<Node>(m_spaces[index++]); });
    }

    template <class Id>
    ExecSpace const& space() const
    {
        std::size_t index = 0, selected = m_spaces.size();
        Graph::for_each_domain([&]<class Node>() {
            if constexpr (std::is_same_v<Id, typename Node::id>)
                selected = index;
            ++index;
        });
        if (selected == m_spaces.size())
            throw std::invalid_argument("execution space requested for a non-physical domain");
        return m_spaces[selected];
    }

    void fence() const
    {
        for (ExecSpace const& space : m_spaces)
            space.fence("multidomain phase complete");
    }
};
} // namespace sil::multidomains
