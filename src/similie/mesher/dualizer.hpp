// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <ddc/ddc.hpp>

namespace sil {

namespace mesher {

struct HalfShifted
{
};

template <class DDim, class Policy = HalfShifted>
struct Dual;

template <class CDim>
struct HalfShiftDualizer;

template <class PrimalDDim>
struct Dual<PrimalDDim, HalfShifted>
    : ddc::UniformPointSampling<typename PrimalDDim::continuous_dimension_type>
{
    using primal_discrete_dimension_type = PrimalDDim;
    using dual_policy = HalfShifted;
};

namespace detail {

template <class CDim, class TypeSeq>
struct DiscreteDimensionForTypeSeq;

template <class CDim, class DDim, class... Tail>
struct DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<DDim, Tail...>>
{
    using type = std::conditional_t<
            std::is_same_v<typename DDim::continuous_dimension_type, CDim>,
            DDim,
            typename DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<Tail...>>::type>;
};

template <class CDim>
struct DiscreteDimensionForTypeSeq<CDim, ddc::detail::TypeSeq<>>
{
    using type = void;
};

template <class CDim, class T>
using discrete_dimension_for_t
        = typename DiscreteDimensionForTypeSeq<CDim, ddc::to_type_seq_t<T>>::type;

template <class Dualizer, class DDim>
struct DualDiscreteDimension;

template <class CDim, class DDim>
struct DualDiscreteDimension<HalfShiftDualizer<CDim>, DDim>
{
    using type = std::conditional_t<
            std::is_same_v<typename DDim::continuous_dimension_type, CDim>,
            Dual<DDim, HalfShifted>,
            DDim>;
};

template <class DDim, class = void>
struct PrimalDiscreteDimension
{
    using type = DDim;
};

template <class DDim>
struct PrimalDiscreteDimension<DDim, std::void_t<typename DDim::primal_discrete_dimension_type>>
{
    using type = typename DDim::primal_discrete_dimension_type;
};

template <class DDim>
using primal_discrete_dimension_t = typename PrimalDiscreteDimension<DDim>::type;

template <class Dualizer, class T>
struct Dualized;

template <class Dualizer, class... DDims>
struct Dualized<Dualizer, ddc::DiscreteElement<DDims...>>
{
    using type = ddc::DiscreteElement<typename DualDiscreteDimension<Dualizer, DDims>::type...>;
};

template <class Dualizer, class... DDims>
struct Dualized<Dualizer, ddc::DiscreteVector<DDims...>>
{
    using type = ddc::DiscreteVector<typename DualDiscreteDimension<Dualizer, DDims>::type...>;
};

template <class Dualizer, class... DDims>
struct Dualized<Dualizer, ddc::DiscreteDomain<DDims...>>
{
    using type = ddc::DiscreteDomain<typename DualDiscreteDimension<Dualizer, DDims>::type...>;
};

template <class Dualizer, class T>
using dualized_t = typename Dualized<Dualizer, T>::type;

template <class T>
struct Primalized;

template <class... DDims>
struct Primalized<ddc::DiscreteElement<DDims...>>
{
    using type = ddc::DiscreteElement<primal_discrete_dimension_t<DDims>...>;
};

template <class... DDims>
struct Primalized<ddc::DiscreteVector<DDims...>>
{
    using type = ddc::DiscreteVector<primal_discrete_dimension_t<DDims>...>;
};

template <class... DDims>
struct Primalized<ddc::DiscreteDomain<DDims...>>
{
    using type = ddc::DiscreteDomain<primal_discrete_dimension_t<DDims>...>;
};

template <class T>
using primalized_t = typename Primalized<T>::type;

template <class TargetDDim, class SourceElem>
KOKKOS_FUNCTION ddc::DiscreteElement<TargetDDim> dualizer_remap_dimension(SourceElem const& source_elem)
{
    using source_d_dim_t
            = discrete_dimension_for_t<typename TargetDDim::continuous_dimension_type, SourceElem>;
    static_assert(!std::is_void_v<source_d_dim_t>);
    return ddc::DiscreteElement<TargetDDim>(ddc::uid<source_d_dim_t>(source_elem));
}

template <class TargetElem, class SourceElem>
struct DualizerRemapElement;

template <class... TargetDDims, class SourceElem>
struct DualizerRemapElement<ddc::DiscreteElement<TargetDDims...>, SourceElem>
{
    KOKKOS_FUNCTION static ddc::DiscreteElement<TargetDDims...> apply(SourceElem const& source_elem)
    {
        return ddc::DiscreteElement<TargetDDims...>(
                dualizer_remap_dimension<TargetDDims>(source_elem)...);
    }
};

template <class TargetElem, class SourceElem>
KOKKOS_FUNCTION TargetElem dualizer_remap_element(SourceElem const& source_elem)
{
    return DualizerRemapElement<TargetElem, SourceElem>::apply(source_elem);
}

template <class Dualizer, class DDim>
auto dualize_subdomain(ddc::DiscreteDomain<DDim> const& subdomain)
{
    using dual_d_dim_t = typename DualDiscreteDimension<Dualizer, DDim>::type;
    if constexpr (std::is_same_v<dual_d_dim_t, DDim>) {
        return subdomain;
    } else {
        static_assert(ddc::is_uniform_point_sampling_v<DDim>);
        auto const nb_points = subdomain.template extent<DDim>().value();
        assert(nb_points > 1);
        auto const lower = ddc::coordinate(subdomain.front())
                           + ddc::Coordinate<typename DDim::continuous_dimension_type>(
                                   ddc::step<DDim>() / 2.);
        auto const upper = ddc::coordinate(subdomain.back())
                           - ddc::Coordinate<typename DDim::continuous_dimension_type>(
                                   ddc::step<DDim>() / 2.);
        return ddc::init_discrete_space<dual_d_dim_t>(dual_d_dim_t::template init<dual_d_dim_t>(
                lower,
                upper,
                ddc::DiscreteVector<dual_d_dim_t>(nb_points - 1)));
    }
}

template <class Dualizer, class T>
struct DualizerApplication;

template <class Dualizer, class... DDims>
struct DualizerApplication<Dualizer, ddc::DiscreteElement<DDims...>>
{
    using input_type = ddc::DiscreteElement<DDims...>;
    using output_type = dualized_t<Dualizer, input_type>;

    KOKKOS_FUNCTION static output_type apply(input_type const& elem)
    {
        return dualizer_remap_element<output_type>(elem);
    }
};

template <class Dualizer, class... DDims>
struct DualizerApplication<Dualizer, ddc::DiscreteVector<DDims...>>
{
    using input_type = ddc::DiscreteVector<DDims...>;
    using output_type = dualized_t<Dualizer, input_type>;

    KOKKOS_FUNCTION static output_type apply(input_type const& vect)
    {
        return output_type(
                ddc::select<DDims>(vect).value()...);
    }
};

template <class Dualizer, class... DDims>
struct DualizerApplication<Dualizer, ddc::DiscreteDomain<DDims...>>
{
    using input_type = ddc::DiscreteDomain<DDims...>;
    using output_type = dualized_t<Dualizer, input_type>;

    static output_type apply(input_type const& dom)
    {
        return output_type(dualize_subdomain<Dualizer>(ddc::DiscreteDomain<DDims>(dom))...);
    }
};

template <class T>
struct PrimalizerApplication;

template <class... DDims>
struct PrimalizerApplication<ddc::DiscreteElement<DDims...>>
{
    using input_type = ddc::DiscreteElement<DDims...>;
    using output_type = primalized_t<input_type>;

    KOKKOS_FUNCTION static output_type apply(input_type const& elem)
    {
        return dualizer_remap_element<output_type>(elem);
    }
};

template <class... DDims>
struct PrimalizerApplication<ddc::DiscreteVector<DDims...>>
{
    using input_type = ddc::DiscreteVector<DDims...>;
    using output_type = primalized_t<input_type>;

    KOKKOS_FUNCTION static output_type apply(input_type const& vect)
    {
        return output_type(
                ddc::select<DDims>(vect).value()...);
    }
};

} // namespace detail

template <class Dualizer, class DDim>
using dual_discrete_dimension_t = typename detail::DualDiscreteDimension<Dualizer, DDim>::type;

template <class CDim>
struct HalfShiftDualizer
{
    template <class T>
    constexpr detail::dualized_t<HalfShiftDualizer<CDim>, T> operator()(T const& obj) const
    {
        return detail::DualizerApplication<HalfShiftDualizer<CDim>, T>::apply(obj);
    }

    template <class T>
    constexpr detail::primalized_t<T> primal(T const& obj) const
    {
        return detail::PrimalizerApplication<T>::apply(obj);
    }
};

} // namespace mesher

} // namespace sil
