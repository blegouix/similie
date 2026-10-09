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
struct A
{
};
struct B
{
};
struct Wall
{
};
struct Value
{
};
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
using Graph = md::Topology<
        ddc::TypeSeq<
                md::Domain<A, Physics, GridX>,
                md::Domain<B, Physics, GridX>,
                md::BoundaryDomain<Wall, sil::exterior::NaturalScalarExtrapolationRule>>,
        md::Connection<
                md::Face<A, GridX, BoundarySide::Upper>,
                md::Face<B, GridX, BoundarySide::Lower>,
                false,
                1.0,
                2.5>,
        md::BoundaryConnection<md::Face<A, GridX, BoundarySide::Lower>, Wall>,
        md::BoundaryConnection<md::Face<B, GridX, BoundarySide::Upper>, Wall>>;

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
    md::Multidomain const
            domains(Graph {},
                    md::domain_data<A>(first, sil::tensor::Tensor(first_device), Physics {}),
                    md::domain_data<B>(second, sil::tensor::Tensor(second_device), Physics {}),
                    md::boundary_data<Wall>(sil::exterior::NaturalScalarExtrapolationRule {}));
    auto const coefficient = [](Physics physics) { return physics.diffusivity; };
    auto const assembled = solvers::assemble_multidomain_laplacian<
            Direction>(Kokkos::DefaultExecutionSpace {}, domains, 128, coefficient, 1.0, normalize);
    auto const data = assemble_matrix_data(assembled.matrix);
    auto const rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), assembled.rhs);
    std::vector<std::map<std::size_t, double>> rows(128);
    for (auto const& entry : data.nonzeros)
        rows[entry.row][entry.column] += entry.value;
    for (std::size_t row = 0; row < 128; ++row) {
        double const diagonal = row == 0 || row == 127 ? 1.0 : 2.0;
        double const scale = normalize ? diagonal : 1.0;
        EXPECT_NEAR(rows[row][row], diagonal / scale, 1.e-12);
        if (row)
            EXPECT_NEAR(rows[row][row - 1], -1.0 / scale, 1.e-12);
        if (row + 1 < 128)
            EXPECT_NEAR(rows[row][row + 1], -1.0 / scale, 1.e-12);
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
        md::DomainExecution<Graph, Kokkos::DefaultExecutionSpace> partitions(
                Kokkos::DefaultExecutionSpace {});
        partitions.for_each([&]<class Node>(Kokkos::DefaultExecutionSpace const& stream) {
            auto const flux = flux_domains.template field<typename Node::id, md::FieldRole::Flux>();
            auto const dual = flux_domains.template extrapolation_rules<
                    typename Node::id,
                    md::FieldRole::Flux>();
            sil::exterior::ScalarLaplacian<
                    Direction,
                    std::remove_cvref_t<decltype(flux)>,
                    std::remove_cvref_t<decltype(dual)>> const op {flux, dual};
            evaluate_rows(stream, domains.template field<typename Node::id>(), op, input, direct);
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
    md::DomainExecution<Graph, Kokkos::DefaultExecutionSpace> execution(
            Kokkos::DefaultExecutionSpace {});
#if defined(KOKKOS_ENABLE_CUDA)
    EXPECT_NE(execution.space<A>().cuda_stream(), execution.space<B>().cuda_stream());
#endif
    EXPECT_THROW(execution.space<Wall>(), std::invalid_argument);
    Kokkos::View<int*> values("stream_values", 2);
    int count = 0;
    execution.for_each([&]<class Node>(Kokkos::DefaultExecutionSpace const& exec) {
        launch_value(exec, values, std::is_same_v<typename Node::id, A> ? 0 : 1);
        ++count;
    });
    execution.fence();
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(host(0), 1);
    EXPECT_EQ(host(1), 2);
}


struct Bottom
{
};
using Graph2D = md::Topology<
        ddc::TypeSeq<
                md::Domain<A, Physics, GridX, GridY>,
                md::BoundaryDomain<Wall, sil::exterior::NaturalScalarExtrapolationRule>,
                md::BoundaryDomain<Value, sil::exterior::PrescribedScalarExtrapolationRule>,
                md::BoundaryDomain<
                        Bottom,
                        sil::exterior::NaturalScalarExtrapolationRule,
                        sil::exterior::NormalScalarFluxExtrapolationRule>>,
        md::BoundaryConnection<md::Face<A, GridX, BoundarySide::Lower>, Wall>,
        md::BoundaryConnection<md::Face<A, GridX, BoundarySide::Upper>, Wall>,
        md::BoundaryConnection<md::Face<A, GridY, BoundarySide::Lower>, Bottom>,
        md::BoundaryConnection<md::Face<A, GridY, BoundarySide::Upper>, Value>>;

void check_2d(double shear, double prescribed_flux, double angle = 0.0)
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
        double const y = elem.uid<GridY>() - 10;
        double const x = elem.uid<GridX>() - 10 + shear * y;
        position(elem, positions.access_element<X>()) = std::cos(angle) * x - std::sin(angle) * y;
        position(elem, positions.access_element<Y>()) = std::sin(angle) * x + std::cos(angle) * y;
        potential.mem(elem, ddc::DiscreteElement<Scalar>(0)) = x * x + y * y;
        std::size_t const id = 5 * (elem.uid<GridX>() - 10) + elem.uid<GridY>() - 10;
        ids(id) = id;
        input_host(id, 0) = x * x + y * y;
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
    md::Multidomain const
            domains(Graph2D {},
                    md::domain_data<A>(field, sil::tensor::Tensor(device_position), Physics {}),
                    md::boundary_data<Wall>(sil::exterior::NaturalScalarExtrapolationRule {}),
                    md::boundary_data<Value>(
                            sil::exterior::PrescribedScalarExtrapolationRule {7.0}),
                    md::boundary_data<Bottom>(
                            sil::exterior::NaturalScalarExtrapolationRule {},
                            sil::exterior::NormalScalarFluxExtrapolationRule {prescribed_flux}));
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
    if (shear == 0.0) {
        EXPECT_NEAR(rows[12][12], 4.0, 1.e-12);
        for (std::size_t neighbor : {7, 11, 13, 17})
            EXPECT_NEAR(rows[12][neighbor], -1.0, 1.e-12);
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
            EXPECT_NEAR(
                    host_applied(row, 0) - rhs(row, 0),
                    reference
                            .mem(ddc::DiscreteElement<GridX, GridY>(x + 10, y + 10),
                                 ddc::DiscreteElement<Scalar>(0)),
                    1.e-10);
        }
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
