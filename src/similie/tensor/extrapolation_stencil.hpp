// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cassert>
#include <limits>

#include <similie/misc/type_seq_conversion.hpp>

#include "owning_tensor.hpp"

namespace sil::tensor {
namespace detail {
template <class Domain, class Memory, class Evaluate>
KOKKOS_FUNCTION auto pointwise_tensor(Evaluate evaluate)
{
    using Accessor = tensor_accessor_for_domain_t<Domain>;
    std::array<double, Accessor::domain().size()> storage {};
    Tensor<double, Domain, Kokkos::layout_right, Memory> tensor(
            ddc::ChunkSpan<
                    double,
                    Domain,
                    Kokkos::layout_right,
                    Memory>(storage.data(), Accessor::domain()));
    ddc::device_for_each(tensor.domain(), [&](auto component) {
        tensor.mem(component) = evaluate(component);
    });
    return OwningTensor(tensor, std::move(storage));
}

template <class Field, class Evaluate>
KOKKOS_FUNCTION auto pointwise_tensor(Evaluate evaluate)
{
    return pointwise_tensor<typename Field::indices_domain_t, typename Field::memory_space>(
            evaluate);
}
} // namespace detail

/** Affine spatial stencil shared by all components of a pointwise tensor.
 * \important This operator and documentation is fully AI-generated.
 * Terms contain domain geometry and logical coordinates, never field values.
 * A donor id distinguishes overlapping local coordinates in connected domains.
 * The maximum size_t id denotes the source domain, independent of its id.
 * evaluate() applies the basis weights to any component-valued function; the
 * constant represents prescribed exterior data or an interface jump.
 */
template <class Domain>
struct ExtrapolationStencil
{
    static constexpr std::size_t rank = ddc::type_seq_size_v<ddc::to_type_seq_t<Domain>>;
    static constexpr std::size_t capacity = (rank + 1) * (rank + 1);
    struct Term
    {
        Domain domain;
        typename Domain::discrete_element_type element;
        std::size_t domain_id = std::numeric_limits<std::size_t>::max();
        double weight = 0.0;
    };
    std::array<Term, capacity> terms {};
    std::size_t count = 0;
    double constant = 0.0;

    template <class Element>
    KOKKOS_FUNCTION void add(
            Domain domain,
            Element element,
            double weight,
            std::size_t id = std::numeric_limits<std::size_t>::max())
    {
        if (weight == 0.0)
            return;
        typename Domain::discrete_element_type stored = [&] {
            if constexpr (requires { element.storage_element(); })
                return typename Domain::discrete_element_type(element.storage_element());
            else
                return typename Domain::discrete_element_type(element);
        }();
        for (std::size_t i = 0; i < count; ++i)
            if (terms[i].domain == domain && terms[i].element == stored
                && terms[i].domain_id == id) {
                terms[i].weight += weight;
                return;
            }
        assert(count < capacity);
        terms[count++] = {domain, stored, id, weight};
    }
    KOKKOS_FUNCTION void accumulate(ExtrapolationStencil const& other, double weight)
    {
        constant += weight * other.constant;
        for (std::size_t i = 0; i < other.count; ++i)
            add(other.terms[i].domain,
                other.terms[i].element,
                weight * other.terms[i].weight,
                other.terms[i].domain_id);
    }
    template <class Evaluate>
    KOKKOS_FUNCTION double evaluate(Evaluate evaluate) const
    {
        double result = constant;
        for (std::size_t i = 0; i < count; ++i)
            result += terms[i].weight * evaluate(terms[i]);
        return result;
    }
};
} // namespace sil::tensor
