// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <concepts>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

// 1D Mesher
template <std::size_t D, class X>
class Mesher1D
{
public:
    static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;

    using bsplines_type = ddc::UniformBSplines<X, D>;

private:
    template <class T>
    using greville_points_type = ddc::GrevilleInterpolationPoints<T, BoundCond, BoundCond>;

public:
    using discrete_dimension_type =
            typename greville_points_type<bsplines_type>::interpolation_discrete_dimension_type;

    template <
            std::derived_from<discrete_dimension_type> DDimX,
            std::derived_from<bsplines_type> BSplinesX>
    ddc::DiscreteDomain<DDimX> mesh(double x_start, double x_end, std::size_t nb_x_points);
};

template <std::size_t D, class X>
template <
        std::derived_from<typename Mesher1D<D, X>::discrete_dimension_type> DDimX,
        std::derived_from<typename Mesher1D<D, X>::bsplines_type> BSplinesX>
ddc::DiscreteDomain<DDimX> Mesher1D<D, X>::mesh(
        double x_start,
        double x_end,
        std::size_t nb_x_points)
{
    ddc::init_discrete_space<
            BSplinesX>(ddc::Coordinate<X>(x_start), ddc::Coordinate<X>(x_end), nb_x_points);
    ddc::init_discrete_space<DDimX>(
            greville_points_type<BSplinesX>::template get_sampling<DDimX>());
    return greville_points_type<BSplinesX>::template get_domain<DDimX>();
}

// ND Mesher
template <std::size_t D, class... X>
class Mesher
{
public:
    static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;

    template <class T>
    using bsplines_type = ddc::UniformBSplines<T, D>;

private:
    template <class T>
    using greville_points_type = ddc::GrevilleInterpolationPoints<T, BoundCond, BoundCond>;

public:
    template <class T>
    using discrete_dimension_type =
            typename greville_points_type<typename Mesher<D, X...>::template bsplines_type<T>>::
                    interpolation_discrete_dimension_type;

    template <class TypeSeqDDimX, class TypeSeqBSplinesX>
    ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDimX> mesh(
            std::array<double, sizeof...(X)> lower_boundaries,
            std::array<double, sizeof...(X)> upper_boundaries,
            std::array<std::size_t, sizeof...(X)> nb_cells);
};

template <std::size_t D, class... X>
template <class TypeSeqDDimX, class TypeSeqBSplinesX>
ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDimX> Mesher<D, X...>::mesh(
        std::array<double, sizeof...(X)> lower_boundaries,
        std::array<double, sizeof...(X)> upper_boundaries,
        std::array<std::size_t, sizeof...(X)> nb_cells)
{
    std::tuple<Mesher1D<D, X>...> meshers;
    std::tuple<ddc::DiscreteDomain<ddc::type_seq_element_t<
            ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>,
            TypeSeqDDimX>>...>
            doms;

    ((std ::get<ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>>(doms)
      = std ::get<ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>>(meshers)
                .template mesh<
                        ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>,
                                TypeSeqDDimX>,
                        ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>,
                                TypeSeqBSplinesX>>(
                        lower_boundaries[ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>],
                        upper_boundaries[ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>],
                        nb_cells[ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>])),
     ...);

    return ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDimX>(
            std::get<ddc::type_seq_rank_v<X, ddc::detail::TypeSeq<X...>>>(doms)...);
}
