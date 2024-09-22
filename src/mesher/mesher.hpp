// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <concepts>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

namespace sil {

namespace mesher {

namespace detail {

// 1D Mesher
template <std::size_t D, class CDim>
class Mesher1D
{
public:
    static constexpr ddc::BoundCond BoundCond = ddc::BoundCond::GREVILLE;

    using bsplines_type = ddc::UniformBSplines<CDim, D>;

private:
    template <class T>
    using greville_points_type = ddc::GrevilleInterpolationPoints<T, BoundCond, BoundCond>;

public:
    using discrete_dimension_type =
            typename greville_points_type<bsplines_type>::interpolation_discrete_dimension_type;

    template <
            std::derived_from<discrete_dimension_type> DDim,
            std::derived_from<bsplines_type> BSplines>
    ddc::DiscreteDomain<DDim> mesh(double x_start, double x_end, std::size_t nb_x_points);
};

template <std::size_t D, class CDim>
template <
        std::derived_from<typename Mesher1D<D, CDim>::discrete_dimension_type> DDim,
        std::derived_from<typename Mesher1D<D, CDim>::bsplines_type> BSplines>
ddc::DiscreteDomain<DDim> Mesher1D<D, CDim>::mesh(
        double x_start,
        double x_end,
        std::size_t nb_x_points)
{
    ddc::init_discrete_space<
            BSplines>(ddc::Coordinate<CDim>(x_start), ddc::Coordinate<CDim>(x_end), nb_x_points);
    ddc::init_discrete_space<DDim>(greville_points_type<BSplines>::template get_sampling<DDim>());
    return greville_points_type<BSplines>::template get_domain<DDim>();
}

} // namespace detail

// ND Mesher
template <std::size_t D, class... CDim>
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
            typename greville_points_type<typename Mesher<D, CDim...>::template bsplines_type<T>>::
                    interpolation_discrete_dimension_type;

    template <class TypeSeqDDim, class TypeSeqBSplines>
    ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim> mesh(
            std::array<double, sizeof...(CDim)> lower_boundaries,
            std::array<double, sizeof...(CDim)> upper_boundaries,
            std::array<std::size_t, sizeof...(CDim)> nb_cells);
};

template <std::size_t D, class... CDim>
template <class TypeSeqDDim, class TypeSeqBSplines>
ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim> Mesher<D, CDim...>::mesh(
        std::array<double, sizeof...(CDim)> lower_boundaries,
        std::array<double, sizeof...(CDim)> upper_boundaries,
        std::array<std::size_t, sizeof...(CDim)> nb_cells)
{
    std::tuple<detail::Mesher1D<D, CDim>...> meshers;
    std::tuple<ddc::DiscreteDomain<ddc::type_seq_element_t<
            ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
            TypeSeqDDim>>...>
            doms;

    ((std ::get<ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>>(doms)
      = std ::get<ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>>(meshers)
                .template mesh<
                        ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
                                TypeSeqDDim>,
                        ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
                                TypeSeqBSplines>>(
                        lower_boundaries[ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>],
                        upper_boundaries[ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>],
                        nb_cells[ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>])),
     ...);

    return ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim>(
            std::get<ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>>(doms)...);
}

} // namespace mesher

} // namespace sil
