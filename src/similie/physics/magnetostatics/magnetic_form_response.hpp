// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <type_traits>

#include <similie/exterior/continuous_hodge_star/continuous_hodge_star.hpp>
#include <similie/exterior/continuous_hodge_star/levi_civita_continuous_hodge_star.hpp>

namespace similie::physics::magnetostatics {

/** Isotropic magnetic response H = nu(|B|_g) star_g B in three dimensions.
 * \important This operator and documentation is fully AI-generated.
 * B contains differential 2-form coefficients (B_yz, B_zx, B_xy), and the
 * output contains 1-form coefficients (H_x, H_y, H_z). No cochain measures
 * enter this operator. The metric and orientation belong to the continuous
 * Hodge star; callers separately reconstruct and reduce their cochains.
 *
 * The supplied equations describe the material energy in an orthonormal
 * frame. Evaluating them along (|B|_g,0,0) provides the radial material law.
 * The metric is Riemannian and the coordinate orientation is positive.
 * This applies to isotropic materials only. The tangent differentiates the
 * same metric norm and Hodge map as the field, including off-diagonal terms.
 */
template <class X, class Y, class Z>
class MagneticFormResponse
{
    struct First : sil::tensor::TensorNaturalIndex<X, Y, Z>
    {
    };
    struct Second : sil::tensor::TensorNaturalIndex<X, Y, Z>
    {
    };
    struct Third : sil::tensor::TensorNaturalIndex<X, Y, Z>
    {
    };

    std::array<double, 9> m_star {};
    double m_volume = 1.0;

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 3> dual(
            std::array<double, 3> const& induction) const
    {
        std::array<double, 3> result {};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                result[row] += m_star[3 * row + column] * induction[column];
        return result;
    }

    [[nodiscard]] KOKKOS_FUNCTION double norm_squared(
            std::array<double, 3> const& induction,
            std::array<double, 3> const& dual_induction) const
    {
        // B wedge star B = |B|_g^2 vol_g in signed complementary order.
        double result = 0.0;
        for (std::size_t component = 0; component < 3; ++component)
            result += induction[component] * dual_induction[component];
        return result / m_volume;
    }

    template <class MemorySpace>
    [[nodiscard]] KOKKOS_FUNCTION static MagneticFormResponse euclidean_impl()
    {
        using MetricIndex = sil::tensor::TensorIdentityIndex<
                sil::tensor::Covariant<sil::tensor::MetricIndex1<X, Y, Z>>,
                sil::tensor::Covariant<sil::tensor::MetricIndex2<X, Y, Z>>>;
        sil::tensor::Tensor metric(
                ddc::ChunkSpan<
                        double,
                        ddc::DiscreteDomain<MetricIndex>,
                        Kokkos::layout_right,
                        MemorySpace>(nullptr, sil::tensor::TensorAccessor<MetricIndex>::domain()));
        return MagneticFormResponse(metric, ddc::DiscreteElement<>());
    }

public:
    [[nodiscard]] KOKKOS_FUNCTION static MagneticFormResponse euclidean()
    {
        KOKKOS_IF_ON_HOST((return euclidean_impl<Kokkos::HostSpace>();))
        KOKKOS_IF_ON_DEVICE(
                (return euclidean_impl<typename Kokkos::DefaultExecutionSpace::memory_space>();))
    }

    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 3> field(
            std::array<double, 3> const& induction,
            double reluctivity) const
    {
        std::array<double, 3> result = dual(induction);
        for (double& component : result)
            component *= reluctivity;
        return result;
    }

    template <class Law, class Element>
    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 3> constitutive_field(
            Law const& law,
            typename Law::State const& state,
            Element const& element) const
    {
        static_assert(Law::N == 3);
        std::array<double, 3> const dual_induction = dual(state.value);
        double const norm = Kokkos::sqrt(norm_squared(state.value, dual_induction));
        typename Law::State radial = state;
        radial.value = {norm, 0.0, 0.0};
        double const reluctivity
                = norm == 0.0 ? law.jacobian(radial, element)[0] : law(radial, element)[0] / norm;
        return field(state.value, reluctivity);
    }

    template <class Metric, class Element>
    KOKKOS_FUNCTION MagneticFormResponse(Metric const& metric, Element const& element)
    {
        using Source = ddc::
                TypeSeq<sil::tensor::Contravariant<First>, sil::tensor::Contravariant<Second>>;
        using Target = ddc::TypeSeq<sil::tensor::Covariant<Third>>;
        using Top = ddc::TypeSeq<
                sil::tensor::Covariant<First>,
                sil::tensor::Covariant<Second>,
                sil::tensor::Covariant<Third>>;
        using MetricIndex
                = ddc::type_seq_element_t<0, ddc::to_type_seq_t<typename Metric::indices_domain_t>>;
        using Star = std::conditional_t<
                sil::misc::Specialization<MetricIndex, sil::tensor::TensorIdentityIndex>,
                sil::exterior::LeviCivitaContinuousHodgeStar<Source, Target, Metric, Element>,
                sil::exterior::ContinuousHodgeStar<Source, Target, Metric, Element>>;
        using Volume = std::conditional_t<
                sil::misc::Specialization<MetricIndex, sil::tensor::TensorIdentityIndex>,
                sil::exterior::LeviCivitaContinuousHodgeStar<ddc::TypeSeq<>, Top, Metric, Element>,
                sil::exterior::ContinuousHodgeStar<ddc::TypeSeq<>, Top, Metric, Element>>;
        auto const star = Star::value(metric, element);
        using SourceIndex = sil::tensor::lower_t<sil::exterior::reduction_index_t<Source>>;
        sil::tensor::TensorAccessor<SourceIndex> accessor;
        std::array<std::size_t, 3> const columns {
                SourceIndex::access_id_to_mem_id(accessor.template access_element<Y, Z>().uid()),
                SourceIndex::access_id_to_mem_id(accessor.template access_element<X, Z>().uid()),
                SourceIndex::access_id_to_mem_id(accessor.template access_element<X, Y>().uid())};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                m_star[3 * row + column]
                        = (column == 1 ? -1.0 : 1.0) * star.weights[3 * row + columns[column]];
        m_volume = Volume::
                value(metric,
                      element,
                      ddc::DiscreteElement<
                              sil::tensor::Covariant<First>,
                              sil::tensor::Covariant<Second>,
                              sil::tensor::Covariant<Third>>(0, 1, 2));
        KOKKOS_ASSERT(m_volume > 0.0);
    }

    template <class Equations, class Element>
    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 3> field(
            Equations const& equations,
            typename Equations::State const& state,
            Element const& element) const
    {
        std::array<double, 3> result = dual(state.moments);
        double const norm = Kokkos::sqrt(norm_squared(state.moments, result));
        typename Equations::State radial = state;
        radial.moments = {norm, 0.0, 0.0};
        double const reluctivity
                = norm == 0.0 ? equations.template jacobian<
                                        std::integral_constant<std::size_t, 0>,
                                        std::integral_constant<std::size_t, 0>>(radial, element)
                              : equations.template dpotential_dt<
                                        std::integral_constant<std::size_t, 0>>(radial, element)
                                        / norm;
        for (double& component : result)
            component *= reluctivity;
        return result;
    }

    template <class Equations, class Element>
    [[nodiscard]] KOKKOS_FUNCTION std::array<double, 9> jacobian(
            Equations const& equations,
            typename Equations::State const& state,
            Element const& element) const
    {
        std::array<double, 3> const dual_induction = dual(state.moments);
        double const q = norm_squared(state.moments, dual_induction);
        double const norm = Kokkos::sqrt(q);
        typename Equations::State radial = state;
        radial.moments = {norm, 0.0, 0.0};
        double const derivative = equations.template jacobian<
                std::integral_constant<std::size_t, 0>,
                std::integral_constant<std::size_t, 0>>(radial, element);
        double const reluctivity
                = norm == 0.0 ? derivative
                              : equations.template dpotential_dt<
                                        std::integral_constant<std::size_t, 0>>(radial, element)
                                        / norm;
        double const radial_derivative = q == 0.0 ? 0.0 : (derivative - reluctivity) / q;
        std::array<double, 9> result {};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                result[3 * row + column] = reluctivity * m_star[3 * row + column]
                                           + radial_derivative / m_volume * dual_induction[row]
                                                     * dual_induction[column];
        return result;
    }
};

} // namespace similie::physics::magnetostatics
