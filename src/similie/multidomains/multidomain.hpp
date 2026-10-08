// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include <similie/exterior/extrapolation_rules.hpp>

#include "execution.hpp"
#include "topology.hpp"

namespace sil::multidomains {

enum class FieldRole { Primal, Flux, Geometry };

/** A non-owning field view carrying the physical graph node's identity. */
template <class Id, class Field>
struct DomainField : Field
{
    using domain_id = Id;
    KOKKOS_FUNCTION explicit DomainField(Field field) : Field(field) {}
};

template <class Id, class Field, class Geometry, class Physics, class Flux = Field>
struct DomainData
{
    using id = Id;
    using physics_type = Physics;
    Field field;
    Geometry geometry;
    Physics physics;
    Flux flux;
};

template <class Id, class Field, class Geometry, class Physics>
auto domain_data(Field field, Geometry geometry, Physics physics)
{
    return DomainData<Id, Field, Geometry, Physics> {field, geometry, physics, field};
}

template <class Id, class Field, class Geometry, class Physics, class Flux>
auto domain_data(Field field, Geometry geometry, Physics physics, Flux flux)
{
    return DomainData<Id, Field, Geometry, Physics, Flux> {field, geometry, physics, flux};
}

template <class Id, class Primal, class Flux, class Geometry>
struct BoundaryData
{
    using id = Id;
    using primal_rule_type = Primal;
    using flux_rule_type = Flux;
    using geometry_rule_type = Geometry;
    Primal primal;
    Flux flux;
    Geometry geometry;
};

template <
        class Id,
        class Primal,
        class Flux = exterior::ZeroCochainExtrapolationRule,
        class Geometry = exterior::ClampCochainExtrapolationRule>
auto boundary_data(Primal primal, Flux flux = {}, Geometry geometry = {})
{
    return BoundaryData<Id, Primal, Flux, Geometry> {primal, flux, geometry};
}

/** Bind fields, geometry, physics and boundary values to a static topology.
 * \important This operator and documentation is fully AI-generated.
 * Data objects are views, not mesh owners. Boundary data has no cochain storage.
 * Factories return ordinary ExtrapolationRules with statically selected donors
 * and source/target faces; the complete graph is never captured in a kernel.
 * value(field, sampler, element, component) forwards mapped DomainField views
 * to the sampler. Its domain_id identifies global stencil columns even where
 * local coordinates overlap. operator() follows exactly the same connections.
 * Primal interface jumps are Connection::JUMP times jump_scale. Reverse rules
 * invert the affine transformation. Geometry never receives a potential jump.
 * Supply a distinct dual cochain to domain_data() when using Flux field rules.
 */
template <class Graph, class... Data>
class Multidomain
{
    std::tuple<Data...> m_data;

    template <class Id>
    static consteval std::size_t data_index()
    {
        static_assert(
                (std::size_t(0) + ... + std::size_t(std::is_same_v<Id, typename Data::id>)) == 1,
                "each topology node needs exactly one data object");
        std::size_t index = 0, result = 0;
        ((std::is_same_v<Id, typename Data::id> ? void(result = index) : void(), ++index), ...);
        return result;
    }

    template <class Node>
    static consteval bool valid_data()
    {
        using State = std::tuple_element_t<data_index<typename Node::id>(), std::tuple<Data...>>;
        if constexpr (Node::IS_BOUNDARY)
            return std::is_same_v<typename Node::primal_rule_type, typename State::primal_rule_type>
                   && std::is_same_v<typename Node::flux_rule_type, typename State::flux_rule_type>
                   && std::is_same_v<
                           typename Node::geometry_rule_type,
                           typename State::geometry_rule_type>;
        else
            return std::is_same_v<typename Node::physics_type, typename State::physics_type>
                   && std::is_same_v<
                           typename Node::dimensions,
                           ddc::to_type_seq_t<
                                   typename decltype(State::field)::non_indices_domain_t>>;
    }

    template <class... Nodes>
    static consteval bool valid_data(ddc::TypeSeq<Nodes...>)
    {
        return sizeof...(Nodes) == sizeof...(Data) && (valid_data<Nodes>() && ...);
    }

    template <class Id, FieldRole Role, class Dimension, exterior::BoundarySide Side>
    auto face_rule(double jump_scale) const
    {
        using SourceFace = Face<Id, Dimension, Side>;
        using Edge = typename Graph::template connection<SourceFace>;
        using Target = std::conditional_t<
                std::is_same_v<SourceFace, typename Edge::first>,
                typename Edge::second,
                typename Edge::first>;
        using TargetNode = typename Graph::template node<typename Target::domain_id>;
        if constexpr (TargetNode::IS_BOUNDARY) {
            if constexpr (Role == FieldRole::Primal)
                return data<typename Target::domain_id>().primal;
            else if constexpr (Role == FieldRole::Flux)
                return data<typename Target::domain_id>().flux;
            else
                return data<typename Target::domain_id>().geometry;
        } else {
            constexpr bool forward = std::is_same_v<SourceFace, typename Edge::first>;
            double const orientation
                    = Role == FieldRole::Geometry
                              ? 1.0
                              : (forward ? Edge::ORIENTATION : 1.0 / Edge::ORIENTATION);
            double const jump
                    = Role == FieldRole::Primal
                              ? jump_scale
                                        * (forward ? Edge::JUMP : -Edge::JUMP / Edge::ORIENTATION)
                              : 0.0;
            return exterior::ExternalDomainExtrapolationRule {
                    field<typename Target::domain_id, Role>(),
                    exterior::ExternalDomainBoundaryMap<
                            Dimension> {Side, Target::SIDE, Edge::SHARED_BOUNDARY},
                    jump,
                    orientation};
        }
    }

    template <class Id, FieldRole Role, class... Dimensions>
    auto rules(double jump_scale, ddc::TypeSeq<Dimensions...>) const
    {
        return exterior::ExtrapolationRules(
                std::
                        pair {face_rule<Id, Role, Dimensions, exterior::BoundarySide::Lower>(
                                      jump_scale),
                              face_rule<Id, Role, Dimensions, exterior::BoundarySide::Upper>(
                                      jump_scale)}...);
    }

public:
    using topology_type = Graph;
    template <class Function>
    static void for_each_domain(Function function)
    {
        Graph::for_each_domain(function);
    }
    /** Submit one domain computation per partition, then join the phase. */
    template <class ExecSpace, class Function>
    void for_each_domain(ExecSpace const& exec, Function function) const
    {
        DomainExecution<Graph, ExecSpace> partitions(exec);
        partitions.for_each(function);
        partitions.fence();
    }

    /** Transform bound state on the host; kernels capture only the resulting face views. */
    template <class Function>
    auto transform_data(Function function) const
    {
        return std::apply(
                [&](auto const&... data) {
                    return Multidomain<
                            Graph,
                            decltype(function(data))...>(Graph {}, function(data)...);
                },
                m_data);
    }

    explicit Multidomain(Graph, Data... data) : m_data(std::move(data)...)
    {
        static_assert(
                valid_data(typename Graph::nodes {}),
                "bound data must match the topology's physics, dimensions and boundary policies");
    }

    template <class Id>
    auto const& data() const
    {
        return std::get<data_index<Id>()>(m_data);
    }

    template <class Id, FieldRole Role = FieldRole::Primal>
    auto field() const
    {
        static_assert(
                !Graph::template node<Id>::IS_BOUNDARY,
                "abstract boundary nodes have no field");
        if constexpr (Role == FieldRole::Primal)
            return DomainField<Id, decltype(data<Id>().field)> {data<Id>().field};
        else if constexpr (Role == FieldRole::Flux)
            return DomainField<Id, decltype(data<Id>().flux)> {data<Id>().flux};
        else
            return DomainField<Id, decltype(data<Id>().geometry)> {data<Id>().geometry};
    }

    template <class Id, FieldRole Role = FieldRole::Primal>
    auto extrapolation_rules(double jump_scale = 1.0) const
    {
        static_assert(!Graph::template node<Id>::IS_BOUNDARY);
        return rules<Id, Role>(jump_scale, typename Graph::template node<Id>::dimensions {});
    }
};

template <class Graph, class... Data>
Multidomain(Graph, Data...) -> Multidomain<Graph, Data...>;
} // namespace sil::multidomains
