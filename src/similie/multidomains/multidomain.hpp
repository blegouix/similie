// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <similie/exterior/extrapolation_rules.hpp>

#include "topology.hpp"

namespace sil::multidomains {

/** One stream per physical domain, retained until all submitted work completes.
 * \important This operator and documentation is fully AI-generated.
 */
template <class ExecSpace>
class DomainExecution
{
    std::vector<ExecSpace> m_spaces;

public:
    DomainExecution(ExecSpace const& parent, std::size_t count)
    {
        if (count == 0)
            throw std::invalid_argument("no physical execution domains");
        parent.fence("multidomain input readiness");
        m_spaces = Kokkos::Experimental::partition_space(parent, std::vector<int>(count, 1));
    }
    template <class Function>
    void for_each(Function function) const
    {
        for (std::size_t id = 0; id < m_spaces.size(); ++id)
            function(id, m_spaces[id]);
    }
    ExecSpace const& space(std::size_t id) const
    {
        return m_spaces.at(id);
    }
    void fence() const
    {
        for (ExecSpace const& space : m_spaces)
            space.fence("multidomain phase complete");
    }
};

enum class FieldRole { Primal, Flux, Geometry };

template <class Field>
struct DomainField : Field
{
    std::size_t domain_id;
    KOKKOS_FUNCTION DomainField(Field field, std::size_t id) : Field(field), domain_id(id) {}
};

/** A thin borrowed donor view, so face policies never duplicate lazy operators.
 * \important This operator and documentation is fully AI-generated.
 */
template <class Field>
struct DomainReference
{
    using non_indices_domain_t = typename Field::non_indices_domain_t;
    using discrete_domain_type = typename Field::discrete_domain_type;
    Field const* field;
    std::size_t domain_id;
    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return field->non_indices_domain();
    }
    template <class Element, class Component>
    KOKKOS_FUNCTION double mem(Element elem, Component component) const
    {
        return field->mem(elem, component);
    }
    template <class Element, class Component>
    KOKKOS_FUNCTION std::size_t global_index(Element elem, Component component) const
    {
        return field->global_index(elem, component);
    }
    template <class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(Sampler sampler, Element elem, Component component) const
    {
        return field->value(sampler, elem, component);
    }
};

template <class Field, class Geometry, class Physics, class Flux = Field>
struct DomainData
{
    using physics_type = Physics;
    Field field;
    Geometry geometry;
    Physics physics;
    Flux flux;
};

template <class Field, class Geometry, class Physics>
auto domain_data(Field field, Geometry geometry, Physics physics)
{
    return DomainData<Field, Geometry, Physics> {field, geometry, physics, field};
}
template <class Field, class Geometry, class Physics, class Flux>
auto domain_data(Field field, Geometry geometry, Physics physics, Flux flux)
{
    return DomainData<Field, Geometry, Physics, Flux> {field, geometry, physics, flux};
}

/** A runtime face policy with the same sampling API as an individual rule.
 * \important This operator and documentation is fully AI-generated.
 * Only this face's donor view and mapping are captured in kernels, never the
 * host adjacency vectors. Runtime flags also guide geometric truncation and
 * Dirichlet assembly. The caller retains ownership of the donor allocation.
 */
template <class Dimension, class Field>
struct FaceRule
{
    exterior::ExternalDomainExtrapolationRule<Dimension, Field> external;
    BoundaryRule boundary;
    bool interface;
    KOKKOS_FUNCTION bool is_interface() const
    {
        return interface;
    }
    KOKKOS_FUNCTION bool is_dirichlet() const
    {
        return !interface && boundary.is_dirichlet();
    }
    template <class Source, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Source const& field, Element elem, Component component) const
    {
        return interface ? external(field, elem, component) : boundary(field, elem, component);
    }
    template <class Source, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            Source const& field,
            Sampler sampler,
            Element elem,
            Component component) const
    {
        return interface ? external.value(field, sampler, elem, component)
                         : boundary.value(field, sampler, elem, component);
    }
    KOKKOS_FUNCTION bool shared_boundary() const
    {
        return external.boundary_map.shared_boundary;
    }
};

/** Bind homogeneous field/geometry/physics views to a runtime topology.
 * \important This operator and documentation is fully AI-generated.
 * Domain count is unrestricted. The topology owns exterior policies; physical
 * data owns views, not mesh allocations. Returned ExtrapolationRules retain the
 * usual per-dimension pairs and normalized outward-distance corner blend.
 * Thin donor references point into host/device-accessible arrays owned by this
 * binding. Keep the binding, source bindings of transformed data, and cochain
 * allocations alive until submitted work finishes. SharedSpace holds metadata
 * only; the cochain allocations retain their original memory space.
 */
template <std::size_t N, class Data>
class Multidomain
{
    template <class Field>
    struct DomainViews
    {
        Kokkos::View<DomainField<Field>*, Kokkos::SharedSpace> values;
        std::size_t constructed = 0;
        DomainViews(std::string const& label, std::size_t count)
            : values(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), count)
        {
        }
        ~DomainViews()
        {
            for (std::size_t id = 0; id < constructed; ++id)
                std::destroy_at(values.data() + id);
        }
    };
    Topology<N> m_topology;
    std::vector<Data> m_data;
    std::shared_ptr<DomainViews<decltype(Data::field)>> m_fields;
    std::shared_ptr<DomainViews<decltype(Data::geometry)>> m_geometry;
    std::shared_ptr<DomainViews<decltype(Data::flux)>> m_flux;

    template <class Field, class Select>
    auto make_views(std::string const& label, Select select) const
    {
        std::shared_ptr<DomainViews<Field>> views
                = std::make_shared<DomainViews<Field>>(label, domain_count());
        for (std::size_t id = 0; id < domain_count(); ++id) {
            std::construct_at(views->values.data() + id, select(m_data[id]), id);
            ++views->constructed;
        }
        return views;
    }
    template <FieldRole Role>
    auto reference(std::size_t id) const
    {
        if constexpr (Role == FieldRole::Primal)
            return DomainReference<
                    DomainField<decltype(Data::field)>> {m_fields->values.data() + id, id};
        else if constexpr (Role == FieldRole::Flux)
            return DomainReference<
                    DomainField<decltype(Data::flux)>> {m_flux->values.data() + id, id};
        else
            return DomainReference<
                    DomainField<decltype(Data::geometry)>> {m_geometry->values.data() + id, id};
    }

    template <FieldRole Role, class Dimension, std::size_t Axis, bool Upper>
    auto face_rule(std::size_t id, double jump_scale) const
    {
        constexpr exterior::BoundarySide side
                = Upper ? exterior::BoundarySide::Upper : exterior::BoundarySide::Lower;
        Connection const& edge = m_topology.connection({id, Axis, side});
        bool const interface = edge.domain < m_data.size();
        auto const donor = reference<Role>(interface ? edge.domain : id);
        BoundaryRule boundary;
        if (!interface) {
            BoundaryDomain const& node = m_topology.boundary(edge.domain);
            if constexpr (Role == FieldRole::Primal)
                boundary = node.primal;
            else if constexpr (Role == FieldRole::Flux)
                boundary = node.flux;
            else
                boundary = node.geometry;
        } else {
            auto const source = field<Role>(id).non_indices_domain();
            auto const target = field<Role>(edge.domain).non_indices_domain();
            for (std::size_t tangent = 0; tangent < N; ++tangent)
                if (tangent != Axis
                    && ddc::detail::array(source.extents())[tangent]
                               != ddc::detail::array(target.extents())[tangent])
                    throw std::invalid_argument("interface tangential extents must match");
        }
        return FaceRule<Dimension, std::remove_cvref_t<decltype(donor)>> {
                {donor,
                 {side, edge.side, edge.shared_boundary},
                 Role == FieldRole::Primal ? jump_scale * edge.jump : 0.0,
                 Role == FieldRole::Geometry ? 1.0 : edge.orientation},
                boundary,
                interface};
    }
    template <FieldRole Role, std::size_t... Axis>
    auto rules(std::size_t id, double scale, std::index_sequence<Axis...>) const
    {
        using Dimensions
                = ddc::to_type_seq_t<typename decltype(field<Role>(id))::non_indices_domain_t>;
        return exterior::ExtrapolationRules(
                std::
                        pair {face_rule<
                                      Role,
                                      ddc::type_seq_element_t<Axis, Dimensions>,
                                      Axis,
                                      false>(id, scale),
                              face_rule<
                                      Role,
                                      ddc::type_seq_element_t<Axis, Dimensions>,
                                      Axis,
                                      true>(id, scale)}...);
    }

public:
    using topology_type = Topology<N>;
    Multidomain(Topology<N> topology, std::vector<Data> data)
        : m_topology(std::move(topology))
        , m_data(std::move(data))
    {
        m_topology.validate();
        if (m_data.size() != m_topology.domain_count())
            throw std::invalid_argument("one state is required per physical domain");
        static_assert(
                N
                == ddc::type_seq_size_v<
                        ddc::to_type_seq_t<typename decltype(Data::field)::non_indices_domain_t>>);
        m_fields = make_views<decltype(Data::field)>("multidomain_fields", [](Data const& state) {
            return state.field;
        });
        m_geometry = make_views<
                decltype(Data::geometry)>("multidomain_geometry", [](Data const& state) {
            return state.geometry;
        });
        m_flux = make_views<decltype(Data::flux)>("multidomain_flux", [](Data const& state) {
            return state.flux;
        });
    }
    Topology<N> const& topology() const
    {
        return m_topology;
    }
    std::size_t domain_count() const
    {
        return m_data.size();
    }
    Data const& data(std::size_t id) const
    {
        return m_data.at(id);
    }
    template <FieldRole Role = FieldRole::Primal>
    auto field(std::size_t id) const
    {
        if constexpr (Role == FieldRole::Primal)
            return DomainField {data(id).field, id};
        else if constexpr (Role == FieldRole::Flux)
            return DomainField {data(id).flux, id};
        else
            return DomainField {data(id).geometry, id};
    }
    template <class Function>
    auto transform_data(Function function) const
    {
        std::vector<std::remove_cvref_t<decltype(function(std::size_t(0), m_data.front()))>> data;
        data.reserve(domain_count());
        for (std::size_t id = 0; id < domain_count(); ++id)
            data.push_back(function(id, m_data[id]));
        return Multidomain<N, typename decltype(data)::value_type>(m_topology, std::move(data));
    }
    template <FieldRole Role = FieldRole::Primal>
    auto extrapolation_rules(std::size_t id, double jump_scale = 1.0) const
    {
        return rules<Role>(id, jump_scale, std::make_index_sequence<N> {});
    }
    template <class ExecSpace, class Function>
    void for_each_domain(ExecSpace const& exec, Function function) const
    {
        DomainExecution<ExecSpace> partitions(exec, domain_count());
        partitions.for_each(function);
        partitions.fence();
    }
};
template <std::size_t N, class Data>
Multidomain(Topology<N>, std::vector<Data>) -> Multidomain<N, Data>;
} // namespace sil::multidomains
