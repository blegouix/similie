// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <array>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <similie/multidomains/multidomains.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor.hpp>

namespace {
namespace md = sil::multidomains;
using sil::tensor::BoundarySide;
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

md::Topology<1> graph(bool shared = false)
{
    md::Topology<1> topology(2);
    std::size_t const wall
            = topology.add_boundary({sil::tensor::NaturalScalarExtrapolationRule {}});
    std::size_t const prescribed
            = topology.add_boundary({sil::tensor::PrescribedScalarExtrapolationRule {7.0}});
    topology.connect({0, 0, BoundarySide::Upper}, {1, 0, BoundarySide::Upper}, shared, 2.0, 3.0);
    topology.connect_boundary({0, 0, BoundarySide::Lower}, wall);
    topology.connect_boundary({1, 0, BoundarySide::Lower}, prescribed);
    return topology;
}
TEST(Multidomains, RuntimeValidation)
{
    EXPECT_NO_THROW(graph().validate());
    md::Topology<1> incomplete(2);
    EXPECT_THROW(incomplete.validate(), std::invalid_argument);
    EXPECT_THROW(
            incomplete.connect({0, 0, BoundarySide::Lower}, {0, 0, BoundarySide::Lower}),
            std::invalid_argument);
    EXPECT_THROW(
            incomplete
                    .connect({0, 0, BoundarySide::Lower}, {1, 0, BoundarySide::Lower}, false, 0.0),
            std::invalid_argument);
    EXPECT_THROW(
            incomplete.connect_boundary({0, 1, BoundarySide::Lower}, 2),
            std::invalid_argument);
    EXPECT_THROW(
            incomplete.connect_boundary({0, 0, BoundarySide::Lower}, 1),
            std::invalid_argument);
    auto complete = graph();
    EXPECT_THROW(
            complete.connect({0, 0, BoundarySide::Lower}, {1, 0, BoundarySide::Lower}),
            std::invalid_argument);
    complete.add_boundary({sil::tensor::ZeroExtrapolationRule {}});
    EXPECT_THROW(complete.validate(), std::invalid_argument);
}

TEST(Multidomains, RuntimeDomainCount)
{
    for (std::size_t count : {3, 7}) {
        md::Topology<1> topology(count);
        std::vector<std::array<double, 3>> storage(count);
        std::vector<md::DomainData<ScalarField<GridX>, ScalarField<GridX>, ScalarPhysics>> data;
        for (std::size_t id = 0; id < count; ++id) {
            topology.connect(
                    {id, 0, BoundarySide::Upper},
                    {(id + 1) % count, 0, BoundarySide::Lower},
                    false);
            storage[id] = {double(3 * id), double(3 * id + 1), double(3 * id + 2)};
            ScalarField<GridX> const
                    field {{ddc::DiscreteElement<GridX>(10 * id), ddc::DiscreteVector<GridX>(3)},
                           storage[id].data()};
            data.push_back(md::domain_data(field, field, ScalarPhysics {1.0}));
        }
        md::Multidomain const domains(topology, std::move(data));
        EXPECT_EQ(domains.domain_count(), count);
        for (std::size_t id = 0; id < count; ++id)
            EXPECT_DOUBLE_EQ(
                    domains.extrapolation_rules(id)(
                            domains.field(id),
                            ddc::DiscreteElement<GridX>(10 * id + 3),
                            ddc::DiscreteElement<typename ScalarField<GridX>::component_type>(0)),
                    storage[(id + 1) % count][0]);
    }
}

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
    md::Topology<sizeof...(Axis)> topology(1);
    for (std::size_t axis = 0; axis < sizeof...(Axis); ++axis)
        topology.connect({0, axis, BoundarySide::Lower}, {0, axis, BoundarySide::Upper}, false);
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
            domains(topology, std::vector {md::domain_data(field, field, ScalarPhysics {1.0})});
    auto const rules = domains.extrapolation_rules(0);
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
            rules.value(domains.field(0), sil::tensor::StoredTensorSampler {}, exterior, component),
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
            domains(graph(),
                    std::
                            vector {md::domain_data(first, first, ScalarPhysics {2.0}),
                                    md::domain_data(second, second, ScalarPhysics {4.0})});
    auto const rules_a = domains.extrapolation_rules(0);
    auto const rules_b = domains.extrapolation_rules(1);
    ddc::DiscreteElement<typename ScalarField<GridX>::component_type> const component(0);
    EXPECT_DOUBLE_EQ(rules_a(first, ddc::DiscreteElement<GridX>(13), component), 15.0);
    EXPECT_DOUBLE_EQ(rules_a(first, ddc::DiscreteElement<GridX>(14), component), 13.0);
    EXPECT_DOUBLE_EQ(rules_b(second, ddc::DiscreteElement<GridX>(33), component), 0.0);
    EXPECT_DOUBLE_EQ(rules_b(second, ddc::DiscreteElement<GridX>(29), component), 7.0);
    EXPECT_DOUBLE_EQ(rules_a(first, ddc::DiscreteElement<GridX>(9), component), 0.0);
    EXPECT_DOUBLE_EQ(domains.data(0).physics.density, 2.0);
    auto const flux = domains.extrapolation_rules<md::FieldRole::Flux>(0, 50.0);
    EXPECT_DOUBLE_EQ(flux(first, ddc::DiscreteElement<GridX>(13), component), 12.0);
    md::Multidomain const
            shared(graph(true),
                   std::
                           vector {md::domain_data(first, first, ScalarPhysics {2.0}),
                                   md::domain_data(second, second, ScalarPhysics {4.0})});
    EXPECT_DOUBLE_EQ(
            shared.extrapolation_rules(0)(first, ddc::DiscreteElement<GridX>(13), component),
            13.0);
    auto const geometry = domains.extrapolation_rules<md::FieldRole::Geometry>(0, 50.0);
    EXPECT_DOUBLE_EQ(geometry(first, ddc::DiscreteElement<GridX>(13), component), 6.0);
    auto basis = [](auto field, auto element, auto) {
        if (field.domain_id == 1)
            return element.template uid<GridX>() == 31 ? 1.0 : 0.0;
        else
            return 0.0;
    };
    EXPECT_DOUBLE_EQ(
            rules_a.value(domains.field(0), basis, ddc::DiscreteElement<GridX>(14), component),
            5.0);
    EXPECT_DOUBLE_EQ(
            rules_a.value(domains.field(0), basis, ddc::DiscreteElement<GridX>(11), component),
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
            domains(graph(),
                    std::
                            vector {md::domain_data(first, first, ScalarPhysics {1.0}),
                                    md::domain_data(second, second, ScalarPhysics {1.0})});
    auto const rules = domains.extrapolation_rules(0);
    auto const field = domains.field(0);
    Kokkos::parallel_for(
            "graph_rules",
            1,
            KOKKOS_LAMBDA(int) {
                ddc::DiscreteElement<typename ScalarField<GridX>::component_type> const component(
                        0);
                result(0) = rules(field, ddc::DiscreteElement<GridX>(13), component);
                result(1) = rules
                                    .value(field,
                                           sil::tensor::StoredTensorSampler {},
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
