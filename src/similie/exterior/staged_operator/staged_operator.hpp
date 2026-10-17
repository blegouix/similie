// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/exterior/concepts.hpp>
#include <similie/exterior/staged_operator/local_operator_storage.hpp>
#include <similie/exterior/staged_operator/spatial_operator_storage.hpp>

namespace sil::exterior {


/** Freeze an operator's geometric coefficients using its selected storage policy.
 * \important This operator and documentation is fully AI-generated.
 * The operator supplies staged_storage_type. The policy builds from value(),
 * owns coefficients, and applies their native local or spatial representation.
 * Rebuild after changing geometry or boundary rules. Local pointwise calls keep
 * the non-staged signatures; geometric arguments there are ignored.
 */
template <class Operator, class Storage = typename Operator::staged_storage_type>
class StagedOperator
{
    using Exec = typename Storage::execution_space;
    Kokkos::View<Exec, Kokkos::HostSpace> m_exec;
    Storage m_storage;

public:
    template <class... Arguments>
        requires requires(Exec const& exec, Arguments... arguments) {
            { Storage::build(exec, arguments...) } -> std::same_as<Storage>;
        }
    StagedOperator(Exec const& exec, Arguments... arguments)
        : m_exec("staged_operator_execution")
        , m_storage(Storage::build(exec, arguments...))
    {
        m_exec() = exec;
    }

    template <class Output, class Input>
        requires StagingStorageConcept<Storage, Output, Input>
    Output operator()(Output output, Input input) const
    {
        return m_storage.apply(m_exec(), output, input);
    }

    template <class Output, class Input, class... Arguments>
        requires(sizeof...(Arguments) > 0 && requires(Storage const& storage, Output output,
                Input input, Arguments... arguments) {
            storage.apply_pointwise(output, input, arguments...);
        })
    KOKKOS_FUNCTION void operator()(Output output, Input input, Arguments... arguments) const
    {
        m_storage.apply_pointwise(output, input, arguments...);
    }

    template <class... Arguments>
        requires requires(Storage const& storage, Arguments... arguments) {
            storage.value(arguments...);
        }
    KOKKOS_FUNCTION auto value(Arguments... arguments) const
    {
        return m_storage.value(arguments...);
    }
};

} // namespace sil::exterior
