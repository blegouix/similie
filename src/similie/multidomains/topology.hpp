// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <similie/tensor/external_domain_extrapolation_rule.hpp>

namespace sil::multidomains {

/** A device-copyable choice of the scalar boundary policies.
 * \important This operator and documentation is fully AI-generated.
 * The selected law and its parameter are runtime data; sampling still invokes
 * the ordinary exterior-library policies. Boundary nodes own no field storage.
 */
class BoundaryRule
{
    enum class Kind { Clamp, Zero, Natural, Prescribed, Flux };
    Kind m_kind = Kind::Clamp;
    double m_value = 0.0;

public:
    BoundaryRule() = default;
    BoundaryRule(tensor::ClampExtrapolationRule) {}
    BoundaryRule(tensor::ZeroExtrapolationRule) : m_kind(Kind::Zero) {}
    BoundaryRule(tensor::NaturalScalarExtrapolationRule) : m_kind(Kind::Natural) {}
    BoundaryRule(tensor::PrescribedScalarExtrapolationRule rule)
        : m_kind(Kind::Prescribed)
        , m_value(rule.prescribed_value)
    {
    }
    BoundaryRule(tensor::ConstantScalarExtrapolationRule rule)
        : m_kind(Kind::Flux)
        , m_value(rule.exterior_value)
    {
    }

    KOKKOS_FUNCTION bool is_dirichlet() const
    {
        return m_kind == Kind::Prescribed;
    }

    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Field const& field, Element elem, Component component) const
    {
        switch (m_kind) {
        case Kind::Zero:
            return tensor::ZeroExtrapolationRule {}(field, elem, component);
        case Kind::Natural:
            return tensor::NaturalScalarExtrapolationRule {}(field, elem, component);
        case Kind::Flux:
            return tensor::ConstantScalarExtrapolationRule {m_value}(field, elem, component);
        case Kind::Prescribed:
            return tensor::PrescribedScalarExtrapolationRule {m_value}(field, elem, component);
        default:
            return tensor::ClampExtrapolationRule {}(field, elem, component);
        }
    }
    template <class Field, class Sampler, class Element, class Component>
    KOKKOS_FUNCTION double value(
            Field const& field,
            Sampler sampler,
            Element elem,
            Component component) const
    {
        return (*this)(tensor::SampledTensor<Field, Sampler> {field, sampler}, elem, component);
    }
};

struct BoundaryDomain
{
    BoundaryRule primal;
    BoundaryRule flux = tensor::ZeroExtrapolationRule {};
    BoundaryRule geometry = tensor::ClampExtrapolationRule {};
};

struct Face
{
    std::size_t domain;
    std::size_t axis;
    tensor::BoundarySide side;
    bool operator==(Face const&) const = default;
};

struct Connection
{
    static constexpr std::size_t UNCONNECTED = std::numeric_limits<std::size_t>::max();
    std::size_t domain = UNCONNECTED;
    tensor::BoundarySide side = tensor::BoundarySide::Lower;
    bool shared_boundary = true;
    double orientation = 1.0;
    double jump = 0.0;
};

/** Runtime adjacency of aligned tensor domains of dimension N.
 * \important This operator and documentation is fully AI-generated.
 * Physical node IDs are [0, domain_count); add_boundary() returns exterior IDs.
 * Each physical face has one connection, to an entry face or an abstract
 * boundary node. connect() installs both directions of the affine relation
 * u_first = orientation*u_second + jump. validate() rejects missing faces and
 * unused boundary nodes. Domain counts, policies and connections are runtime
 * data. Ordered dimension tags and matching tangential extents remain required;
 * this graph does not interpolate nonconforming interfaces or rotate frames.
 */
template <std::size_t N>
class Topology
{
    static_assert(N > 0);
    std::vector<std::array<Connection, 2 * N>> m_faces;
    std::vector<BoundaryDomain> m_boundaries;

    Connection& slot(Face face)
    {
        if (face.domain >= m_faces.size() || face.axis >= N)
            throw std::invalid_argument("invalid physical face");
        return m_faces[face.domain][2 * face.axis + (face.side == tensor::BoundarySide::Upper)];
    }

public:
    explicit Topology(std::size_t domains) : m_faces(domains)
    {
        if (domains == 0)
            throw std::invalid_argument("a multidomain needs a physical domain");
    }
    std::size_t domain_count() const
    {
        return m_faces.size();
    }
    std::size_t add_boundary(BoundaryDomain boundary)
    {
        m_boundaries.push_back(boundary);
        return domain_count() + m_boundaries.size() - 1;
    }
    BoundaryDomain const& boundary(std::size_t id) const
    {
        return m_boundaries.at(id - domain_count());
    }
    void connect(
            Face first,
            Face second,
            bool shared_boundary = true,
            double orientation = 1.0,
            double jump = 0.0)
    {
        Connection& a = slot(first);
        Connection& b = slot(second);
        if (first == second || first.axis != second.axis || !std::isfinite(orientation)
            || orientation == 0.0 || !std::isfinite(jump) || !std::isfinite(1.0 / orientation)
            || !std::isfinite(jump / orientation) || a.domain != Connection::UNCONNECTED
            || b.domain != Connection::UNCONNECTED)
            throw std::invalid_argument("invalid or duplicate interface connection");
        a = {second.domain, second.side, shared_boundary, orientation, jump};
        b = {first.domain, first.side, shared_boundary, 1.0 / orientation, -jump / orientation};
    }
    void connect_boundary(Face face, std::size_t id)
    {
        Connection& entry = slot(face);
        if (id < domain_count() || id - domain_count() >= m_boundaries.size()
            || entry.domain != Connection::UNCONNECTED)
            throw std::invalid_argument("invalid or duplicate boundary connection");
        entry.domain = id;
    }
    Connection const& connection(Face face) const
    {
        if (face.axis >= N)
            throw std::invalid_argument("invalid face axis");
        return m_faces.at(face.domain)[2 * face.axis + (face.side == tensor::BoundarySide::Upper)];
    }
    void validate() const
    {
        std::vector<bool> used(m_boundaries.size(), false);
        for (std::array<Connection, 2 * N> const& faces : m_faces)
            for (Connection const& edge : faces) {
                if (edge.domain == Connection::UNCONNECTED)
                    throw std::invalid_argument("every physical face needs a connection");
                if (edge.domain >= domain_count())
                    used.at(edge.domain - domain_count()) = true;
            }
        for (bool active : used)
            if (!active)
                throw std::invalid_argument("unconnected boundary node");
    }
};
} // namespace sil::multidomains
