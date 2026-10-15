// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <cmath>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>
#include <similie/exterior/coboundary.hpp>

namespace {
struct X
{
};
struct Y
{
};
struct Z
{
};
struct W
{
};
struct Spectator
{
};
struct OtherDirection : sil::tensor::TensorNaturalIndex<X, Y, Z, W>
{
};
struct ThirdDirection : sil::tensor::TensorNaturalIndex<X, Y, Z, W>
{
};
struct FourthDirection : sil::tensor::TensorNaturalIndex<X, Y, Z, W>
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
struct GridZ
{
    using continuous_dimension_type = Z;
};
struct GridW
{
    using continuous_dimension_type = W;
};
struct GridSpectator
{
    using continuous_dimension_type = Spectator;
};

template <class Direction, class Cochain, class... Grid>
void check_incidence(ddc::DiscreteDomain<Grid...>)
{
    using Output = sil::exterior::coboundary_index_t<Direction, Cochain>;
    using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<double**> values("incidence_reference", 4, Output::mem_size());
    auto const chain
            = sil::exterior::tangent_basis<Cochain::rank() + 1, ddc::DiscreteDomain<Grid...>>(
                    Kokkos::DefaultExecutionSpace());
    auto const lower = sil::exterior::tangent_basis<Cochain::rank(), ddc::DiscreteDomain<Grid...>>(
            Kokkos::DefaultExecutionSpace());
    Kokkos::parallel_for(
            "compare_incidence_with_boundary",
            1,
            KOKKOS_LAMBDA(int) {
                ddc::DiscreteElement<GridSpectator, Grid...> const
                        element(7, (static_cast<void>(sizeof(Grid)), 4)...);
                auto const evaluate = [](auto sample, auto component) {
                    double phase = 0.13 * sample.template uid<GridSpectator>()
                                   + 0.31 * component.template uid<Cochain>();
                    std::size_t axis = 0;
                    ((phase += (0.17 + 0.11 * axis++) * sample.template uid<Grid>()), ...);
                    return Kokkos::sin(phase);
                };
                auto forward = sil::exterior::detail::make_stencil<MemorySpace, Output>(
                        ddc::DiscreteElement<>());
                auto transposed = sil::exterior::detail::make_stencil<MemorySpace, Output>(
                        ddc::DiscreteElement<>());
                sil::exterior::Coboundary<Direction, Cochain>::operator()(
                        sil::tensor::Tensor(forward.span_view()),
                        evaluate,
                        chain,
                        lower,
                        element);
                sil::exterior::TransposedCoboundary<Direction, Cochain>::operator()(
                        sil::tensor::Tensor(transposed.span_view()),
                        evaluate,
                        chain,
                        lower,
                        element);
                for (std::size_t c = 0; c < chain.size(); ++c) {
                    typename decltype(chain)::simplex_type const
                            simplex(std::integral_constant<std::size_t, Cochain::rank() + 1> {},
                                    ddc::DiscreteElement<Grid...>(element),
                                    chain[c].discrete_vector());
                    auto const faces = sil::exterior::boundary<MemorySpace>(simplex);
                    double expected_forward = 0.0, expected_transposed = 0.0;
                    for (std::size_t f = 0; f < faces.size(); ++f) {
                        std::size_t source = 0;
                        while (lower[source].discrete_vector() != faces[f].discrete_vector())
                            ++source;
                        ddc::DiscreteElement<Cochain> const component(source);
                        double const sign = faces[f].negative() ? -1.0 : 1.0;
                        expected_forward
                                += sign
                                   * evaluate(
                                           ddc::DiscreteElement<GridSpectator, Grid...>(
                                                   ddc::DiscreteElement<GridSpectator>(element),
                                                   faces[f].discrete_element()),
                                           component);
                        ddc::DiscreteElement<Grid...> sample = faces[f].discrete_element();
                        if (f >= Cochain::rank() + 1) {
                            for (std::size_t axis = 0; axis < sizeof...(Grid); ++axis)
                                if (ddc::detail::array(simplex.discrete_vector())[axis] != 0
                                    && ddc::detail::array(faces[f].discrete_vector())[axis] == 0) {
                                    ddc::detail::array(sample)[axis] -= 2;
                                    break;
                                }
                        }
                        expected_transposed
                                -= sign
                                   * evaluate(
                                           ddc::DiscreteElement<GridSpectator, Grid...>(
                                                   ddc::DiscreteElement<GridSpectator>(element),
                                                   sample),
                                           component);
                    }
                    values(0, c) = forward.mem(ddc::DiscreteElement<Output>(c));
                    values(1, c) = expected_forward;
                    values(2, c) = transposed.mem(ddc::DiscreteElement<Output>(c));
                    values(3, c) = expected_transposed;
                }
            });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    for (std::size_t c = 0; c < Output::mem_size(); ++c) {
        EXPECT_NEAR(host(0, c), host(1, c), 1.e-14);
        EXPECT_NEAR(host(2, c), host(3, c), 1.e-14);
    }
}

TEST(Coboundary, ApplicationMatchesBoundaryInFourDimensions)
{
    using Direction = sil::tensor::Covariant<sil::tensor::TensorNaturalIndex<X, Y, Z, W>>;
    ddc::DiscreteDomain<GridX, GridY, GridZ, GridW> const domain {};
    check_incidence<Direction, sil::tensor::Covariant<sil::tensor::ScalarIndex>>(domain);
    check_incidence<Direction, sil::tensor::Covariant<OtherDirection>>(domain);
    check_incidence<
            Direction,
            sil::tensor::TensorAntisymmetricIndex<
                    sil::tensor::Covariant<OtherDirection>,
                    sil::tensor::Covariant<ThirdDirection>>>(domain);
    check_incidence<
            Direction,
            sil::tensor::TensorAntisymmetricIndex<
                    sil::tensor::Covariant<OtherDirection>,
                    sil::tensor::Covariant<ThirdDirection>,
                    sil::tensor::Covariant<FourthDirection>>>(domain);
}
} // namespace
