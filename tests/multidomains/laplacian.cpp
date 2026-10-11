// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <map>
#include <vector>

#include <gtest/gtest.h>
#include <similie/solvers/multidomain_laplacian.hpp>

namespace {
namespace md = sil::multidomains;
namespace solvers = similie::solvers;
using sil::exterior::BoundarySide;
struct X
{
};
struct Y
{
};
struct GridX
{
    using continuous_dimension_type = X;
};
struct GridY
{
    using continuous_dimension_type = Y;
};
struct Physics
{
    double diffusivity = 1.0;
};
md::Topology<1> split_graph()
{
    md::Topology<1> graph(2);
    std::size_t const wall = graph.add_boundary({sil::exterior::NaturalScalarExtrapolationRule {}});
    graph.connect({0, 0, BoundarySide::Upper}, {1, 0, BoundarySide::Lower}, false, 1.0, 2.5);
    graph.connect_boundary({0, 0, BoundarySide::Lower}, wall);
    graph.connect_boundary({1, 0, BoundarySide::Upper}, wall);
    return graph;
}

struct GlobalSampler
{
    Kokkos::View<double**> values;
    template <class Field, class Element, class Component>
    KOKKOS_FUNCTION double operator()(Field field, Element elem, Component component) const
    {
        return values(field.global_index(elem, component), 0);
    }
};

template <class Field, class Operator>
void evaluate_rows(
        Kokkos::DefaultExecutionSpace const& exec,
        Field field,
        Operator op,
        Kokkos::View<double**> input,
        Kokkos::View<double**> output)
{
    ddc::parallel_for_each(
            "evaluate_dec_rows",
            exec,
            field.non_indices_domain(),
            KOKKOS_LAMBDA(typename Field::non_indices_domain_t::discrete_element_type elem) {
                output(field.global_index(
                               elem,
                               ddc::DiscreteElement<typename Field::component_type>(0)),
                       0)
                        = op.value(GlobalSampler {input}, elem);
            });
}

void check_split(bool normalize)
{
    using Direction = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X>>;
    using Position = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X>>;
    using Grid = ddc::DiscreteDomain<GridX>;
    using PositionDomain = ddc::DiscreteDomain<GridX, Position>;
    Grid const first_grid(ddc::DiscreteElement<GridX>(10), ddc::DiscreteVector<GridX>(64)),
            second_grid(ddc::DiscreteElement<GridX>(200), ddc::DiscreteVector<GridX>(64));
    sil::tensor::TensorAccessor<Position> accessor;
    ddc::Chunk
            first_host(PositionDomain(first_grid, accessor.domain()), ddc::HostAllocator<double>()),
            second_host(
                    PositionDomain(second_grid, accessor.domain()),
                    ddc::HostAllocator<double>());
    sil::tensor::Tensor first_position(first_host), second_position(second_host);
    ddc::host_for_each(first_grid, [&](ddc::DiscreteElement<GridX> elem) {
        first_position(elem, accessor.access_element<X>()) = elem.uid<GridX>() - 10;
    });
    ddc::host_for_each(second_grid, [&](ddc::DiscreteElement<GridX> elem) {
        second_position(elem, accessor.access_element<X>()) = elem.uid<GridX>() - 200 + 64;
    });
    Kokkos::View<std::size_t*, Kokkos::HostSpace> first_numbering("first", 64),
            second_numbering("second", 64);
    for (std::size_t i = 0; i < 64; ++i) {
        first_numbering(i) = i;
        second_numbering(i) = 64 + i;
    }
    ddc::Chunk first_device(
            PositionDomain(first_grid, accessor.domain()),
            ddc::DeviceAllocator<double>()),
            second_device(
                    PositionDomain(second_grid, accessor.domain()),
                    ddc::DeviceAllocator<double>());
    ddc::parallel_deepcopy(first_device, first_host);
    ddc::parallel_deepcopy(second_device, second_host);
    Kokkos::View<std::size_t*> first_ids("first_device_ids", 64),
            second_ids("second_device_ids", 64);
    Kokkos::deep_copy(first_ids, first_numbering);
    Kokkos::deep_copy(second_ids, second_numbering);
    solvers::IndexedScalarField<GridX> const first {first_grid, first_ids.data()},
            second {second_grid, second_ids.data()};
    md::Multidomain const domains(
            split_graph(),
            std::vector {
                    md::domain_data(first, sil::tensor::Tensor(first_device), Physics {}),
                    md::domain_data(second, sil::tensor::Tensor(second_device), Physics {})});
    auto const coefficient = [](Physics physics) { return physics.diffusivity; };
    auto const assembled = solvers::assemble_multidomain_laplacian<
            Direction>(Kokkos::DefaultExecutionSpace {}, domains, 128, coefficient, 1.0, normalize);
    auto const data = assemble_matrix_data(assembled.matrix);
    auto const rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), assembled.rhs);
    std::vector<std::map<std::size_t, double>> rows(128);
    for (auto const& entry : data.nonzeros)
        rows[entry.row][entry.column] += entry.value;
    for (std::size_t row = 0; row < 128; ++row) {
        double const diagonal = 2.0;
        double const neighbor = row == 0 || row == 127 ? 2.0 : 1.0;
        double const scale = normalize ? diagonal : 1.0;
        EXPECT_NEAR(rows[row][row], diagonal / scale, 1.e-12);
        if (row)
            EXPECT_NEAR(rows[row][row - 1], -neighbor / scale, 1.e-12);
        if (row + 1 < 128)
            EXPECT_NEAR(rows[row][row + 1], -neighbor / scale, 1.e-12);
        EXPECT_NEAR(rhs(row, 0), (row == 63 ? 2.5 : (row == 64 ? -2.5 : 0.0)) / scale, 1.e-12);
        for (auto const& [column, value] : rows[row])
            if (column + 1 < row || column > row + 1)
                EXPECT_NEAR(value, 0.0, 1.e-12);
    }
    if (!normalize) {
        Kokkos::View<double**> input("probe", 128, 1), applied("csr", 128, 1),
                direct("dec", 128, 1);
        auto host_input = Kokkos::create_mirror_view(input);
        for (std::size_t i = 0; i < 128; ++i)
            host_input(i, 0) = std::sin(0.13 * i);
        Kokkos::deep_copy(input, host_input);
        assembled.matrix.apply(Kokkos::DefaultExecutionSpace {}, input, applied);
        auto const flux_domains
                = solvers::make_laplacian_flux_domains<Direction>(domains, coefficient);
        md::DomainExecution<Kokkos::DefaultExecutionSpace>
                partitions(Kokkos::DefaultExecutionSpace {}, domains.domain_count());
        partitions.for_each([&](std::size_t id, Kokkos::DefaultExecutionSpace const& stream) {
            auto const flux = flux_domains.template field<md::FieldRole::Flux>(id);
            auto const dual = flux_domains.template extrapolation_rules<md::FieldRole::Flux>(id);
            sil::exterior::ScalarLaplacian<
                    Direction,
                    std::remove_cvref_t<decltype(flux)>,
                    std::remove_cvref_t<decltype(dual)>> const op {flux, dual};
            evaluate_rows(stream, domains.field(id), op, input, direct);
        });
        partitions.fence();
        auto const host_applied = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), applied);
        auto const host_direct = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), direct);
        for (std::size_t i = 0; i < 128; ++i)
            EXPECT_NEAR(host_applied(i, 0) - rhs(i, 0), host_direct(i, 0), 1.e-11);
    }
}
TEST(MultidomainsLaplacian, DecInterfaceJump)
{
    check_split(false);
}
TEST(MultidomainsLaplacian, NormalizedDecInterfaceJump)
{
    check_split(true);
}

void launch_value(Kokkos::DefaultExecutionSpace const& exec, Kokkos::View<int*> values, int index)
{
    Kokkos::parallel_for(
            "domain_stream_value",
            Kokkos::RangePolicy(exec, 0, 1),
            KOKKOS_LAMBDA(int) { values(index) = index + 1; });
}
TEST(MultidomainsLaplacian, OneExecutionInstancePerPhysicalDomain)
{
    md::DomainExecution<Kokkos::DefaultExecutionSpace>
            execution(Kokkos::DefaultExecutionSpace {}, 2);
#if defined(KOKKOS_ENABLE_CUDA)
    EXPECT_NE(execution.space(0).cuda_stream(), execution.space(1).cuda_stream());
#endif
    EXPECT_THROW(execution.space(2), std::out_of_range);
    Kokkos::View<int*> values("stream_values", 2);
    int count = 0;
    execution.for_each([&](std::size_t id, Kokkos::DefaultExecutionSpace const& exec) {
        launch_value(exec, values, static_cast<int>(id));
        ++count;
    });
    execution.fence();
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(host(0), 1);
    EXPECT_EQ(host(1), 2);
}



void check_2d(
        double shear,
        double prescribed_flux,
        double angle = 0.0,
        double spacing = 1.0,
        int polynomial = 0,
        double warp = 0.0)
{
    using Direction = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y>>;
    using Scalar = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
    using Position = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;
    using Metric = sil::tensor::TensorIdentityIndex<
            sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y>>,
            sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y>>>;
    using Grid = ddc::DiscreteDomain<GridX, GridY>;
    Grid const
            grid(ddc::DiscreteElement<GridX, GridY>(10, 10),
                 ddc::DiscreteVector<GridX, GridY>(5, 5));
    sil::tensor::TensorAccessor<Position> positions;
    [[maybe_unused]] sil::tensor::TensorAccessor<Scalar> scalars;
    [[maybe_unused]] sil::tensor::TensorAccessor<Metric> metrics;
    ddc::Chunk host_position(
            ddc::DiscreteDomain<GridX, GridY, Position>(grid, positions.domain()),
            ddc::HostAllocator<double>());
    ddc::Chunk host_potential(
            ddc::DiscreteDomain<GridX, GridY, Scalar>(grid, scalars.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor position(host_position);
    sil::tensor::Tensor potential(host_potential);
    Kokkos::View<std::size_t*, Kokkos::HostSpace> ids("host_ids", 25);
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::HostSpace>
            input_host("input_host", 25, 1);
    ddc::host_for_each(grid, [&](ddc::DiscreteElement<GridX, GridY> elem) {
        double const y = spacing * (elem.uid<GridY>() - 10);
        double const x = spacing * (elem.uid<GridX>() - 10) * (1.0 + warp * y) + shear * y;
        double const value = polynomial == 0 ? x * x + y * y : (polynomial == 1 ? x * y : x);
        position(elem, positions.access_element<X>()) = std::cos(angle) * x - std::sin(angle) * y;
        position(elem, positions.access_element<Y>()) = std::sin(angle) * x + std::cos(angle) * y;
        potential.mem(elem, ddc::DiscreteElement<Scalar>(0)) = value;
        std::size_t const id = 5 * (elem.uid<GridX>() - 10) + elem.uid<GridY>() - 10;
        ids(id) = id;
        input_host(id, 0) = value;
    });
    ddc::Chunk device_position(host_position.domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk device_potential(host_potential.domain(), ddc::DeviceAllocator<double>()),
            output(host_potential.domain(), ddc::DeviceAllocator<double>());
    ddc::Chunk
            metric(ddc::DiscreteDomain<GridX, GridY, Metric>(grid, metrics.domain()),
                   ddc::DeviceAllocator<double>());
    ddc::parallel_deepcopy(device_position, host_position);
    ddc::parallel_deepcopy(device_potential, host_potential);
    Kokkos::View<std::size_t*> device_ids("device_ids", 25);
    Kokkos::deep_copy(device_ids, ids);
    solvers::IndexedScalarField<GridX, GridY> const field {grid, device_ids.data()};
    md::Topology<2> graph(1);
    std::size_t const wall = graph.add_boundary({sil::exterior::NaturalScalarExtrapolationRule {}});
    std::size_t const value
            = graph.add_boundary({sil::exterior::PrescribedScalarExtrapolationRule {7.0}});
    std::size_t const bottom = graph.add_boundary(
            {sil::exterior::NaturalScalarExtrapolationRule {},
             sil::exterior::NormalScalarFluxExtrapolationRule {prescribed_flux}});
    graph.connect_boundary({0, 0, BoundarySide::Lower}, wall);
    graph.connect_boundary({0, 0, BoundarySide::Upper}, wall);
    graph.connect_boundary({0, 1, BoundarySide::Lower}, bottom);
    graph.connect_boundary({0, 1, BoundarySide::Upper}, value);
    md::Multidomain const domains(
            graph,
            std::vector {md::domain_data(field, sil::tensor::Tensor(device_position), Physics {})});
    auto const coefficient = [](Physics physics) { return physics.diffusivity; };
    auto const system = solvers::assemble_multidomain_laplacian<
            Direction>(Kokkos::DefaultExecutionSpace {}, domains, 25, coefficient, 1.0, false);
    auto const data = assemble_matrix_data(system.matrix);
    auto const rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), system.rhs);
    std::vector<std::map<std::size_t, double>> rows(25);
    for (auto const& entry : data.nonzeros)
        rows[entry.row][entry.column] += entry.value;
    for (std::size_t x = 0; x < 5; ++x) {
        std::size_t const row = 5 * x + 4;
        EXPECT_EQ(rows[row].size(), 1);
        EXPECT_DOUBLE_EQ(rows[row][row], 1.0);
        EXPECT_DOUBLE_EQ(rhs(row, 0), 7.0);
    }
    if (shear == 0.0 && warp == 0.0) {
        EXPECT_NEAR(rows[12][12], 4.0 / (spacing * spacing), 1.e-10);
        for (std::size_t neighbor : {7, 11, 13, 17})
            EXPECT_NEAR(rows[12][neighbor], -1.0 / (spacing * spacing), 1.e-10);
        EXPECT_NEAR(rows[12][6], 0.0, 1.e-12); // DEC axial stencil, no FEM diagonal coupling.
    }
    if (prescribed_flux != 0.0)
        EXPECT_GT(std::abs(rhs(10, 0)), 0.1);
    Kokkos::View<double**> input("input", 25, 1), applied("applied", 25, 1);
    Kokkos::deep_copy(input, input_host);
    system.matrix.apply(Kokkos::DefaultExecutionSpace {}, input, applied);
    auto const host_applied = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), applied);
    auto staged = sil::exterior::make_staged_laplacian<Metric, Direction, Scalar>(
            Kokkos::DefaultExecutionSpace {},
            sil::tensor::Tensor(output),
            sil::tensor::Tensor(device_potential),
            sil::tensor::Tensor(metric),
            sil::tensor::Tensor(device_position));
    staged(sil::tensor::Tensor(output),
           sil::tensor::Tensor(device_potential),
           sil::exterior::NaturalScalarExtrapolationRule {},
           sil::exterior::ZeroCochainExtrapolationRule {});
    auto output_host = ddc::
            create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace {}, output.span_view());
    sil::tensor::Tensor reference(output_host);
    for (std::size_t x = 1; x < 4; ++x)
        for (std::size_t y = 1; y < 4; ++y) {
            std::size_t const row = 5 * x + y;
            EXPECT_NEAR(host_applied(row, 0) - rhs(row, 0), polynomial == 0 ? -4.0 : 0.0, 1.e-10);
            if (warp == 0.0)
                EXPECT_NEAR(
                        host_applied(row, 0) - rhs(row, 0),
                        reference
                                .mem(ddc::DiscreteElement<GridX, GridY>(x + 10, y + 10),
                                     ddc::DiscreteElement<Scalar>(0)),
                        1.e-10);
        }
    if (polynomial == 2 && prescribed_flux == 0.0)
        for (std::size_t x = 1; x < 4; ++x)
            EXPECT_NEAR(host_applied(5 * x, 0) - rhs(5 * x, 0), 0.0, 1.e-10);
    if (shear == 0.3 && spacing == 0.5 && polynomial == 2 && warp == 0.3) {
        // The geometric Hodge accepts stored one-cochains independently of d.
        [[maybe_unused]] sil::tensor::TensorAccessor<Direction> gradient_accessor;
        ddc::Chunk gradient(
                ddc::DiscreteDomain<GridX, GridY, Direction>(grid, gradient_accessor.domain()),
                ddc::DeviceAllocator<double>());
        sil::exterior::deriv<Direction, Scalar>(
                Kokkos::DefaultExecutionSpace {},
                sil::tensor::Tensor(gradient),
                sil::tensor::Tensor(device_potential),
                sil::exterior::NaturalScalarExtrapolationRule {});
        auto const hodge = sil::exterior::make_tensor_dual_hodge<Direction>(
                sil::tensor::Tensor(gradient),
                sil::tensor::Tensor(device_position),
                sil::exterior::NaturalScalarExtrapolationRule {},
                sil::exterior::ZeroCochainExtrapolationRule {});
        auto const dual = sil::exterior::
                make_extrapolation_rules(hodge, sil::exterior::ZeroCochainExtrapolationRule {});
        sil::exterior::Codifferential<
                Direction,
                std::remove_cvref_t<decltype(hodge)>,
                std::remove_cvref_t<decltype(dual)>> const delta {hodge, dual};
        Kokkos::View<double*> composed("delta_d", 1);
        Kokkos::parallel_for(
                "stored_one_cochain_hodge",
                1,
                KOKKOS_LAMBDA(int) {
                    composed(0) = delta(ddc::DiscreteElement<GridX, GridY>(12, 12));
                });
        auto const result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), composed);
        EXPECT_NEAR(result(0), 0.0, 1.e-10);
    }
}
TEST(MultidomainsLaplacian, SkewGridAnalyticLaplacianUnderRefinement)
{
    for (double spacing : {1.0, 0.5, 0.25})
        for (double shear : {-0.7, 0.3, 1.0})
            for (int polynomial : {0, 1, 2})
                check_2d(shear, 0.0, 0.37, spacing, polynomial);
}
struct Z
{
};
struct GridZ
{
    using continuous_dimension_type = Z;
};
TEST(MultidomainsLaplacian, ConstantPhysicalGradientOnWarpedCells)
{
    for (double spacing : {1.0, 0.5, 0.25})
        for (double warp : {-0.1, 0.3})
            check_2d(0.3, 0.0, 0.37, spacing, 2, warp);
}

TEST(MultidomainsLaplacian, ThreeDimensionalSkewGrid)
{
    using Direction = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y, Z>>;
    using Position = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y, Z>>;
    using Grid = ddc::DiscreteDomain<GridX, GridY, GridZ>;
    Grid const
            grid(ddc::DiscreteElement<GridX, GridY, GridZ>(0, 0, 0),
                 ddc::DiscreteVector<GridX, GridY, GridZ>(5, 5, 5));
    sil::tensor::TensorAccessor<Position> accessor;
    ddc::Chunk host_position(
            ddc::DiscreteDomain<GridX, GridY, GridZ, Position>(grid, accessor.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor position(host_position);
    Kokkos::View<std::size_t*, Kokkos::HostSpace> ids("ids", 125);
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace::array_layout, Kokkos::HostSpace>
            input_host("input", 125, 1);
    ddc::host_for_each(grid, [&](ddc::DiscreteElement<GridX, GridY, GridZ> elem) {
        double const x
                = 0.5 * (elem.uid<GridX>() + 0.4 * elem.uid<GridY>() + 0.2 * elem.uid<GridZ>());
        double const y = 0.5 * (elem.uid<GridY>() + 0.3 * elem.uid<GridZ>());
        double const z = 0.5 * elem.uid<GridZ>();
        position(elem, accessor.access_element<X>()) = x;
        position(elem, accessor.access_element<Y>()) = y;
        position(elem, accessor.access_element<Z>()) = z;
        std::size_t const id = 25 * elem.uid<GridX>() + 5 * elem.uid<GridY>() + elem.uid<GridZ>();
        ids(id) = id;
        input_host(id, 0) = x * y + x * x + y * y + z * z;
    });
    ddc::Chunk device_position(host_position.domain(), ddc::DeviceAllocator<double>());
    ddc::parallel_deepcopy(device_position, host_position);
    Kokkos::View<std::size_t*> device_ids("ids", 125);
    Kokkos::deep_copy(device_ids, ids);
    solvers::IndexedScalarField<GridX, GridY, GridZ> const field {grid, device_ids.data()};
    md::Topology<3> graph(1);
    std::size_t const wall = graph.add_boundary({sil::exterior::NaturalScalarExtrapolationRule {}});
    for (std::size_t axis = 0; axis < 3; ++axis) {
        graph.connect_boundary({0, axis, BoundarySide::Lower}, wall);
        graph.connect_boundary({0, axis, BoundarySide::Upper}, wall);
    }
    md::Multidomain const domains(
            graph,
            std::vector {md::domain_data(field, sil::tensor::Tensor(device_position), Physics {})});
    auto const system = solvers::assemble_multidomain_laplacian<Direction>(
            Kokkos::DefaultExecutionSpace {},
            domains,
            125,
            [](Physics physics) { return physics.diffusivity; },
            1.0,
            false);
    Kokkos::View<double**> input("input", 125, 1), output("output", 125, 1);
    Kokkos::deep_copy(input, input_host);
    system.matrix.apply(Kokkos::DefaultExecutionSpace {}, input, output);
    auto const result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
    for (std::size_t i = 1; i < 4; ++i)
        for (std::size_t j = 1; j < 4; ++j)
            for (std::size_t k = 1; k < 4; ++k)
                EXPECT_NEAR(result(25 * i + 5 * j + k, 0), -6.0, 1.e-10);
}
TEST(MultidomainsLaplacian, DecAxialStencilAndDirichlet)
{
    check_2d(0.0, 0.0);
}
TEST(MultidomainsLaplacian, MappedDecAgreesWithStagedLaplacian)
{
    check_2d(0.3, 0.0);
}
TEST(MultidomainsLaplacian, PrescribedDualCochainFlux)
{
    check_2d(0.0, 1.0);
}
} // namespace

TEST(MultidomainsLaplacian, RotatedDecAxialStencil)
{
    // Include a quarter turn: Cartesian diagonal projections then vanish,
    // although the cell Jacobian and the DEC Hodge star remain nonsingular.
    for (double angle : {0.37, std::acos(-1.0) / 2.0, 2.1}) {
        check_2d(0.0, 0.0, angle);
        check_2d(0.3, 0.0, angle);
    }
}
