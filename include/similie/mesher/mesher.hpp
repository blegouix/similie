// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

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
    constexpr ddc::DiscreteDomain<DDim> mesh(
            ddc::Coordinate<CDim> lower_boundary,
            ddc::Coordinate<CDim> upper_boundary,
            ddc::DiscreteVector<DDim> nb_cells)
    {
        ddc::init_discrete_space<BSplines>(lower_boundary, upper_boundary, nb_cells);
        ddc::init_discrete_space<DDim>(
                greville_points_type<BSplines>::template get_sampling<DDim>());
        return greville_points_type<BSplines>::template get_domain<DDim>();
    };
};

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
    constexpr ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim> mesh(
            ddc::Coordinate<CDim...> lower_boundaries,
            ddc::Coordinate<CDim...> upper_boundaries,
            ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim>::mlength_type nb_cells);
};

template <std::size_t D, class... CDim>
template <class TypeSeqDDim, class TypeSeqBSplines>
constexpr ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim> Mesher<D, CDim...>::mesh(
        ddc::Coordinate<CDim...> lower_boundaries,
        ddc::Coordinate<CDim...> upper_boundaries,
        ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim>::mlength_type nb_cells)
{
    std::tuple<detail::Mesher1D<D, CDim>...> meshers;
    std::tuple<ddc::DiscreteDomain<ddc::type_seq_element_t<
            ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
            TypeSeqDDim>>...>
            meshs;

    ((std ::get<ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>>(meshs)
      = std ::get<ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>>(meshers)
                .template mesh<
                        ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
                                TypeSeqDDim>,
                        ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
                                TypeSeqBSplines>>(
                        ddc::select<CDim>(lower_boundaries),
                        ddc::select<CDim>(upper_boundaries),
                        ddc::select<ddc::type_seq_element_t<
                                ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>,
                                TypeSeqDDim>>(nb_cells))),
     ...);

    return ddc::detail::convert_type_seq_to_discrete_domain_t<TypeSeqDDim>(
            std::get<ddc::type_seq_rank_v<CDim, ddc::detail::TypeSeq<CDim...>>>(meshs)...);
}

} // namespace mesher

} // namespace sil
