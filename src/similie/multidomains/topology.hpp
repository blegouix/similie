// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <similie/exterior/external_domain_extrapolation_rule.hpp>

namespace sil::multidomains {

/** A physical graph node: identity, physics, and ordered logical dimensions.
 * \important This documentation is fully AI-generated.
 * The topology owns no mesh or cochain storage. Bind those separately in a
 * Multidomain. Every face must have exactly one connection, including exterior
 * faces. Exterior connections terminate in abstract BoundaryDomain nodes.
 */
template <class Id, class Physics, class... Dimensions>
struct Domain
{
    using id = Id;
    using physics_type = Physics;
    using dimensions = ddc::TypeSeq<Dimensions...>;
    static constexpr bool IS_BOUNDARY = false;
};

/** An abstract exterior node, with no mesh, unknowns, or volume equations.
 * \important This documentation is fully AI-generated.
 * The three policies describe primal values, prescribed integrated outward
 * flux, and geometry sampling. Their values are supplied when binding data.
 */
template <
        class Id,
        class PrimalRule,
        class FluxRule = exterior::ZeroCochainExtrapolationRule,
        class GeometryRule = exterior::ClampCochainExtrapolationRule>
struct BoundaryDomain
{
    using id = Id;
    using primal_rule_type = PrimalRule;
    using flux_rule_type = FluxRule;
    using geometry_rule_type = GeometryRule;
    static constexpr bool IS_BOUNDARY = true;
};

template <class DomainId, class Dimension, exterior::BoundarySide Side>
struct Face
{
    using domain_id = DomainId;
    using dimension = Dimension;
    static constexpr exterior::BoundarySide SIDE = Side;
};

template <class BoundaryId>
struct BoundaryEndpoint
{
    using domain_id = BoundaryId;
    using dimension = void;
};

/** An undirected connection, declaring both entry faces once.
 * \important This documentation is fully AI-generated.
 * In the first-to-second direction, u_first = Orientation*u_second + Jump*scale.
 * The reverse rule automatically uses 1/Orientation and -Jump/Orientation.
 * SharedBoundary distinguishes coincident trace nodes from disjoint ownership.
 * Aligned interfaces have the same normal tag and ordered tangential tags.
 * Boundary endpoints must use the default identity transform.
 */
template <
        class First,
        class Second,
        bool SharedBoundary = true,
        double Orientation = 1.0,
        double Jump = 0.0>
struct Connection
{
    using first = First;
    using second = Second;
    static constexpr bool SHARED_BOUNDARY = SharedBoundary;
    static constexpr double ORIENTATION = Orientation;
    static constexpr double JUMP = Jump;
};

template <class PhysicalFace, class BoundaryId>
using BoundaryConnection = Connection<PhysicalFace, BoundaryEndpoint<BoundaryId>>;

namespace detail {

template <class Id, class... Nodes>
struct FindNode
{
    using type = void;
};

template <class Id, class Node, class... Nodes>
struct FindNode<Id, Node, Nodes...>
{
    using type = std::conditional_t<
            std::is_same_v<Id, typename Node::id>,
            Node,
            typename FindNode<Id, Nodes...>::type>;
};

template <class Endpoint, class... Nodes>
consteval bool valid_endpoint()
{
    using Node = typename FindNode<typename Endpoint::domain_id, Nodes...>::type;
    if constexpr (std::is_void_v<Node>)
        return false;
    else if constexpr (Node::IS_BOUNDARY)
        return std::is_void_v<typename Endpoint::dimension>;
    else
        return ddc::type_seq_contains_v<
                ddc::TypeSeq<typename Endpoint::dimension>,
                typename Node::dimensions>;
}

template <class FaceType, class... Edges>
inline constexpr std::size_t face_degree
        = (std::size_t(0) + ...
           + (std::size_t(std::is_same_v<FaceType, typename Edges::first>)
              + std::size_t(std::is_same_v<FaceType, typename Edges::second>)));

template <class FaceType, class... Edges>
struct FindConnection
{
    using type = void;
};

template <class FaceType, class Edge, class... Edges>
struct FindConnection<FaceType, Edge, Edges...>
{
    using type = std::conditional_t<
            std::is_same_v<FaceType, typename Edge::first>
                    || std::is_same_v<FaceType, typename Edge::second>,
            Edge,
            typename FindConnection<FaceType, Edges...>::type>;
};

template <class Nodes, class... Edges>
struct TopologyValidation;

template <class... Nodes, class... Edges>
struct TopologyValidation<ddc::TypeSeq<Nodes...>, Edges...>
{
    template <class Edge>
    static consteval bool valid_edge()
    {
        if constexpr (
                !valid_endpoint<typename Edge::first, Nodes...>()
                || !valid_endpoint<typename Edge::second, Nodes...>())
            return false;
        else {
            using First = typename FindNode<typename Edge::first::domain_id, Nodes...>::type;
            using Second = typename FindNode<typename Edge::second::domain_id, Nodes...>::type;
            if constexpr (First::IS_BOUNDARY && Second::IS_BOUNDARY)
                return false;
            else if constexpr (First::IS_BOUNDARY || Second::IS_BOUNDARY)
                return Edge::ORIENTATION == 1.0 && Edge::JUMP == 0.0;
            else
                return std::is_same_v<typename First::dimensions, typename Second::dimensions>
                       && std::is_same_v<
                               typename Edge::first::dimension,
                               typename Edge::second::dimension>
                       && Edge::ORIENTATION != 0.0;
        }
    }

    template <class Node, class... Dimensions>
    static consteval bool covered(ddc::TypeSeq<Dimensions...>)
    {
        return ((face_degree<
                         Face<typename Node::id, Dimensions, exterior::BoundarySide::Lower>,
                         Edges...>
                         == 1
                 && face_degree<
                            Face<typename Node::id, Dimensions, exterior::BoundarySide::Upper>,
                            Edges...>
                            == 1)
                && ...);
    }

    template <class Node>
    static consteval bool valid_node()
    {
        if (((std::size_t(std::is_same_v<typename Node::id, typename Nodes::id>)) + ...) != 1)
            return false;
        if constexpr (Node::IS_BOUNDARY)
            return (std::size_t(0) + ...
                    + (std::size_t(
                               std::is_same_v<typename Node::id, typename Edges::first::domain_id>)
                       + std::size_t(
                               std::is_same_v<
                                       typename Node::id,
                                       typename Edges::second::domain_id>)))
                   > 0;
        else
            return ddc::type_seq_size_v<typename Node::dimensions> > 0
                   && covered<Node>(typename Node::dimensions {});
    }

    static constexpr bool VALID
            = sizeof...(Nodes) > 0 && (valid_edge<Edges>() && ...) && (valid_node<Nodes>() && ...);
};
} // namespace detail

/** Check topology completeness without instantiating an invalid Topology. */
template <class Nodes, class... Connections>
inline constexpr bool is_valid_topology_v
        = detail::TopologyValidation<Nodes, Connections...>::VALID;

/** A compile-time graph: no implicit exterior faces and no runtime adjacency.
 * \important This documentation is fully AI-generated.
 * Connections may form cycles or self-periodic domains. Boundary nodes may be
 * shared by several faces but cannot connect to each other. All physical faces
 * have degree one; dangling domains and duplicate face connections are rejected.
 */
template <class Nodes, class... Connections>
struct Topology;

template <class... Nodes, class... Connections>
struct Topology<ddc::TypeSeq<Nodes...>, Connections...>
{
    static_assert(
            is_valid_topology_v<ddc::TypeSeq<Nodes...>, Connections...>,
            "invalid multidomain topology: every physical face needs exactly one valid connection "
            "to a domain or boundary node");
    using nodes = ddc::TypeSeq<Nodes...>;
    using connections = ddc::TypeSeq<Connections...>;
    template <class Id>
    using node = typename detail::FindNode<Id, Nodes...>::type;
    template <class FaceType>
    using connection = typename detail::FindConnection<FaceType, Connections...>::type;

    template <class Function>
    static void for_each_domain(Function function)
    {
        (
                [&]() {
                    if constexpr (!Nodes::IS_BOUNDARY)
                        function.template operator()<Nodes>();
                }(),
                ...);
    }
};
} // namespace sil::multidomains
