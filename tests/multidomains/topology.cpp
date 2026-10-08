// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <similie/multidomains/multidomains.hpp>
#include <similie/solvers/affine_scalar_system.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor.hpp>

namespace {
namespace md = sil::multidomains;
using sil::exterior::BoundarySide;
struct A
{
};
struct B
{
};
struct Wall
{
};
struct Unknown
{
};
struct Prescribed
{
};
struct X
{
};
struct GridX
{
    using continuous_dimension_type = X;
};
struct ScalarPhysics
{
    double density;
};

template <class... Dimensions>
struct ScalarField
{
    using component_type = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using non_indices_domain_t = ddc::DiscreteDomain<Dimensions...>;
    using discrete_domain_type = ddc::DiscreteDomain<Dimensions..., component_type>;
    non_indices_domain_t grid;
    double* storage;
    KOKKOS_FUNCTION non_indices_domain_t non_indices_domain() const
    {
        return grid;
    }
    template <class Element, class Component>
    KOKKOS_FUNCTION double mem(Element element, Component) const
    {
        std::size_t index = 0;
        ((index = index * grid.template extent<Dimensions>() + element.template uid<Dimensions>()
                  - grid.front().template uid<Dimensions>()),
         ...);
        return storage[index];
    }
};

using Nodes = ddc::TypeSeq<
        md::Domain<A, ScalarPhysics, GridX>,
        md::Domain<B, ScalarPhysics, GridX>,
        md::BoundaryDomain<Wall, sil::exterior::NaturalScalarExtrapolationRule>,
        md::BoundaryDomain<Prescribed, sil::exterior::PrescribedScalarExtrapolationRule>>;
using Interface = md::Connection<
        md::Face<A, GridX, BoundarySide::Upper>,
        md::Face<B, GridX, BoundarySide::Upper>,
        false,
        2.0,
        3.0>;
using LeftWall = md::BoundaryConnection<md::Face<A, GridX, BoundarySide::Lower>, Wall>;
using RightValue = md::BoundaryConnection<md::Face<B, GridX, BoundarySide::Lower>, Prescribed>;
using Graph = md::Topology<Nodes, Interface, LeftWall, RightValue>;

static_assert(md::is_valid_topology_v<Nodes, Interface, LeftWall, RightValue>);
static_assert(!md::is_valid_topology_v<Nodes, Interface, LeftWall>);
static_assert(!md::is_valid_topology_v<Nodes, Interface, LeftWall, RightValue, RightValue>);
static_assert(!md::is_valid_topology_v<
              Nodes,
              Interface,
              LeftWall,
              md::BoundaryConnection<md::Face<B, GridX, BoundarySide::Lower>, A>>);
static_assert(!md::is_valid_topology_v<
              Nodes,
              Interface,
              LeftWall,
              RightValue,
              md::Connection<md::BoundaryEndpoint<Wall>, md::BoundaryEndpoint<Prescribed>>>);


static_assert(!md::is_valid_topology_v<
              Nodes,
              Interface,
              LeftWall,
              md::BoundaryConnection<md::Face<B, GridX, BoundarySide::Lower>, Unknown>>);
static_assert(!md::is_valid_topology_v<
              Nodes,
              Interface,
              LeftWall,
              RightValue,
              md::BoundaryConnection<md::Face<A, X, BoundarySide::Lower>, Wall>>);
static_assert(!md::is_valid_topology_v<
              ddc::TypeSeq<
                      md::Domain<A, ScalarPhysics, GridX>,
                      md::BoundaryDomain<Wall, sil::exterior::NaturalScalarExtrapolationRule>,
                      md::BoundaryDomain<
                              Prescribed,
                              sil::exterior::PrescribedScalarExtrapolationRule>>,
              LeftWall,
              md::BoundaryConnection<md::Face<A, GridX, BoundarySide::Upper>, Wall>>);

template <std::size_t Axis>
struct Coordinate
{
};
template <std::size_t Axis>
struct GridDimension
{
    using continuous_dimension_type = Coordinate<Axis>;
};

template <std::size_t... Axis>
void check_periodic_rank(std::index_sequence<Axis...>)
{
    using PeriodicGraph = md::Topology<
            ddc::TypeSeq<md::Domain<A, ScalarPhysics, GridDimension<Axis>...>>,
            md::Connection<
                    md::Face<A, GridDimension<Axis>, BoundarySide::Lower>,
                    md::Face<A, GridDimension<Axis>, BoundarySide::Upper>,
                    false>...>;
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < sizeof...(Axis); ++axis)
        count *= 3;
    std::vector<double> storage(count);
    for (std::size_t index = 0; index < count; ++index)
        storage[index] = index;
    ScalarField<GridDimension<Axis>...> const
            field {{ddc::DiscreteElement<GridDimension<Axis>...>((void(Axis), 0)...),
                    ddc::DiscreteVector<GridDimension<Axis>...>((void(Axis), 3)...)},
                   storage.data()};
    md::Multidomain const
            domains(PeriodicGraph {}, md::domain_data<A>(field, field, ScalarPhysics {1.0}));
    auto const rules = domains.template extrapolation_rules<A>();
    ddc::DiscreteElement<GridDimension<Axis>...> exterior((void(Axis), 1)...);
    exterior.template uid<GridDimension<0>>() = std::size_t(-1);
    exterior.template uid<GridDimension<1>>() = 4;
    ddc::DiscreteElement<GridDimension<Axis>...> first((void(Axis), 1)...), second(first);
    first.template uid<GridDimension<0>>() = 2;
    first.template uid<GridDimension<1>>() = 2;
    second.template uid<GridDimension<0>>() = 0;
    second.template uid<GridDimension<1>>() = 1;
    ddc::DiscreteElement<typename decltype(field)::component_type> const component(0);
    double const expected
            = (field.mem(first, component) + 2.0 * field.mem(second, component)) / 3.0;
    EXPECT_DOUBLE_EQ(rules(field, exterior, component), expected);
    EXPECT_DOUBLE_EQ(
            rules
                    .value(domains.template field<A>(),
                           sil::exterior::StoredCochainSampler {},
                           exterior,
                           component),
            expected);
}

TEST(Multidomains, PeriodicCorners2DAnd4D)
{
    check_periodic_rank(std::make_index_sequence<2> {});
    check_periodic_rank(std::make_index_sequence<4> {});
}


TEST(Multidomains, EntrySideAffineInverseAndPhysics)
{
    std::array<double, 3> a {1, 2, 3}, b {4, 5, 6};
    ScalarField<GridX> const
            first {{ddc::DiscreteElement<GridX>(10), ddc::DiscreteVector<GridX>(3)}, a.data()};
    ScalarField<GridX> const
            second {{ddc::DiscreteElement<GridX>(30), ddc::DiscreteVector<GridX>(3)}, b.data()};
    md::Multidomain const
            domains(Graph {},
                    md::domain_data<A>(first, first, ScalarPhysics {2.0}),
                    md::domain_data<B>(second, second, ScalarPhysics {4.0}),
                    md::boundary_data<Wall>(sil::exterior::NaturalScalarExtrapolationRule {}),
                    md::boundary_data<Prescribed>(
                            sil::exterior::PrescribedScalarExtrapolationRule {7.0}));
    auto const rules_a = domains.extrapolation_rules<A>();
    auto const rules_b = domains.extrapolation_rules<B>();
    ddc::DiscreteElement<typename ScalarField<GridX>::component_type> const component(0);
    EXPECT_DOUBLE_EQ(rules_a(first, ddc::DiscreteElement<GridX>(13), component), 15.0);
    EXPECT_DOUBLE_EQ(rules_a(first, ddc::DiscreteElement<GridX>(14), component), 13.0);
    EXPECT_DOUBLE_EQ(rules_b(second, ddc::DiscreteElement<GridX>(33), component), 0.0);
    EXPECT_DOUBLE_EQ(rules_b(second, ddc::DiscreteElement<GridX>(29), component), 7.0);
    EXPECT_DOUBLE_EQ(rules_a(first, ddc::DiscreteElement<GridX>(9), component), 0.0);
    EXPECT_DOUBLE_EQ(domains.data<A>().physics.density, 2.0);
    auto const flux = domains.extrapolation_rules<A, md::FieldRole::Flux>(50.0);
    EXPECT_DOUBLE_EQ(flux(first, ddc::DiscreteElement<GridX>(13), component), 12.0);
    using SharedGraph = md::Topology<
            Nodes,
            md::Connection<
                    md::Face<A, GridX, BoundarySide::Upper>,
                    md::Face<B, GridX, BoundarySide::Upper>,
                    true,
                    2.0,
                    3.0>,
            LeftWall,
            RightValue>;
    md::Multidomain const
            shared(SharedGraph {},
                   md::domain_data<A>(first, first, ScalarPhysics {2.0}),
                   md::domain_data<B>(second, second, ScalarPhysics {4.0}),
                   md::boundary_data<Wall>(sil::exterior::NaturalScalarExtrapolationRule {}),
                   md::boundary_data<Prescribed>(
                           sil::exterior::PrescribedScalarExtrapolationRule {7.0}));
    EXPECT_DOUBLE_EQ(
            shared.extrapolation_rules<A>()(first, ddc::DiscreteElement<GridX>(13), component),
            13.0);
    auto const geometry = domains.extrapolation_rules<A, md::FieldRole::Geometry>(50.0);
    EXPECT_DOUBLE_EQ(geometry(first, ddc::DiscreteElement<GridX>(13), component), 6.0);
    auto basis = [](auto field, auto element, auto) {
        if constexpr (std::is_same_v<typename decltype(field)::domain_id, B>)
            return element.template uid<GridX>() == 31 ? 1.0 : 0.0;
        else
            return 0.0;
    };
    EXPECT_DOUBLE_EQ(
            rules_a.value(domains.field<A>(), basis, ddc::DiscreteElement<GridX>(14), component),
            5.0);
    EXPECT_DOUBLE_EQ(
            rules_a.value(domains.field<A>(), basis, ddc::DiscreteElement<GridX>(11), component),
            0.0);
}

void check_generated_rules_on_cuda()
{
    Kokkos::View<double*> a("a", 3), b("b", 3), result("result", 2);
    Kokkos::deep_copy(a, 1.0);
    Kokkos::deep_copy(b, 4.0);
    ScalarField<GridX> const
            first {{ddc::DiscreteElement<GridX>(10), ddc::DiscreteVector<GridX>(3)}, a.data()};
    ScalarField<GridX> const
            second {{ddc::DiscreteElement<GridX>(30), ddc::DiscreteVector<GridX>(3)}, b.data()};
    md::Multidomain const
            domains(Graph {},
                    md::domain_data<A>(first, first, ScalarPhysics {1.0}),
                    md::domain_data<B>(second, second, ScalarPhysics {1.0}),
                    md::boundary_data<Wall>(sil::exterior::NaturalScalarExtrapolationRule {}),
                    md::boundary_data<Prescribed>(
                            sil::exterior::PrescribedScalarExtrapolationRule {7.0}));
    auto const rules = domains.extrapolation_rules<A>();
    auto const field = domains.field<A>();
    Kokkos::parallel_for(
            "graph_rules",
            1,
            KOKKOS_LAMBDA(int) {
                ddc::DiscreteElement<typename ScalarField<GridX>::component_type> const component(
                        0);
                result(0) = rules(field, ddc::DiscreteElement<GridX>(13), component);
                result(1) = rules
                                    .value(field,
                                           sil::exterior::StoredCochainSampler {},
                                           ddc::DiscreteElement<GridX>(13),
                                           component);
            });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    EXPECT_DOUBLE_EQ(host(0), 11.0);
    EXPECT_DOUBLE_EQ(host(1), host(0));
}
TEST(Multidomains, GeneratedRulesOnCuda)
{
    check_generated_rules_on_cuda();
}

} // namespace
