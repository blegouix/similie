// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/tensor/tensor.hpp>

namespace sil::multidomains {
/** Index-only scalar cochain; sampling maps local nodes to global unknowns.
 * \important This operator and documentation is fully AI-generated.
 * The caller owns the index array in the selected execution space. Nodes use
 * disjoint ownership across domains, in DDC LayoutRight order.
 */
template <class... DDim>
struct IndexedScalarField
{
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    using component_type = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using non_indices_domain_t = ddc::DiscreteDomain<DDim...>;
    using discrete_domain_type = ddc::DiscreteDomain<DDim..., component_type>;
    using indices_domain_t = ddc::DiscreteDomain<component_type>;

    using accessor_t = sil::tensor::TensorAccessor<component_type>;
    using discrete_element_type = typename discrete_domain_type::discrete_element_type;
    using natural_domain_t = discrete_domain_type;
    using element_type = double;
    KOKKOS_FUNCTION static constexpr accessor_t accessor()
    {
        return accessor_t();
    }
    KOKKOS_FUNCTION indices_domain_t indices_domain() const
    {
        return accessor().domain();
    }
    KOKKOS_FUNCTION discrete_domain_type domain() const
    {
        return discrete_domain_type(grid, indices_domain());
    }
    KOKKOS_FUNCTION discrete_domain_type access_domain() const
    {
        return domain();
    }
    KOKKOS_FUNCTION natural_domain_t natural_domain() const
    {
        return domain();
    }

    non_indices_domain_t grid;
    std::size_t const* indices;

    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return grid;
    }

    template <class Element, class Component>
    KOKKOS_FUNCTION std::size_t global_index(Element elem, Component) const
    {
        std::size_t local = 0;
        ((local = local * grid.template extent<DDim>() + elem.template uid<DDim>()
                  - grid.front().template uid<DDim>()),
         ...);
        assert(sil::misc::domain_contains(grid, elem));
        return indices[local];
    }
};


} // namespace sil::multidomains
