// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <array>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <similie/misc/domain_contains.hpp>
#include <similie/tensor/character.hpp>
#include <similie/tensor/tensor.hpp>

#include "coboundary.hpp"
#include "local_chain.hpp"
#include "reduction_and_reconstruction.hpp"

namespace sil::exterior {

namespace detail {

template <class Elem>
struct ElementSpatialDomain;

template <class... Tags>
struct ElementSpatialDomain<ddc::DiscreteElement<Tags...>>
{
    using type = ddc::DiscreteDomain<Tags...>;
};

} // namespace detail

struct ZeroConnection
{
    template <class OutputComponentIndex, class DerivativeIndex, class InputComponentIndex>
    [[nodiscard]] KOKKOS_FUNCTION double value(auto, auto) const
    {
        return 0.0;
    }
};

template <class... SpatialIndex>
class CovariantDerivative
{
    using spatial_index_seq = ddc::detail::TypeSeq<SpatialIndex...>;
    using gradient_form_index = tensor::Covariant<tensor::TensorNaturalIndex<SpatialIndex...>>;
    using gradient_index_seq
            = tensor::upper_t<ddc::to_type_seq_t<tensor::natural_domain_t<gradient_form_index>>>;
    using scalar_component_index = tensor::ScalarIndex;

public:
    template <
            class OutputComponentIndex,
            class DerivativeIndex,
            class InputComponentIndex = OutputComponentIndex,
            class Elem,
            class PositionType,
            class Connection = ZeroConnection>
    [[nodiscard]] KOKKOS_FUNCTION static auto value(
            Elem elem,
            PositionType position,
            Connection connection = ZeroConnection {})
    {
        using first_spatial_index = ddc::type_seq_element_t<0, spatial_index_seq>;
        auto stencil = reduced_derivative_stencil_<first_spatial_index>(elem);
        stencil *= 0.0;
        if constexpr (std::is_same_v<OutputComponentIndex, InputComponentIndex>) {
            ((stencil += reconstructed_value_<SpatialIndex, DerivativeIndex>(elem, position)), ...);
        }
        double const connection_coefficient = connection.template value<
                OutputComponentIndex,
                DerivativeIndex,
                InputComponentIndex>(elem, position);
        auto const center_elem = typename decltype(stencil)::
                discrete_element_type(elem, ddc::DiscreteElement<scalar_component_index>(0));
        if (misc::domain_contains(stencil.domain(), center_elem)) {
            stencil.mem(center_elem) += connection_coefficient;
        }
        return stencil;
    }

    template <
            class OutputComponentIndex,
            class DerivativeIndex,
            class InputComponentIndex = OutputComponentIndex,
            class Elem,
            class PositionType,
            class Connection = ZeroConnection>
    [[nodiscard]] KOKKOS_FUNCTION auto operator()(
            Elem elem,
            PositionType position,
            Connection connection = ZeroConnection {}) const
    {
        return value<
                OutputComponentIndex,
                DerivativeIndex,
                InputComponentIndex>(elem, position, connection);
    }

private:
    template <class DerivativeIndex, class Elem>
    [[nodiscard]] KOKKOS_FUNCTION static auto reduced_derivative_stencil_(Elem elem)
    {
        using spatial_domain_type = typename detail::ElementSpatialDomain<Elem>::type;

        auto const chain = tangent_basis<1, spatial_domain_type>(elem);
        auto const lower_chain = tangent_basis<0, spatial_domain_type>(elem);
        [[maybe_unused]] tensor::TensorAccessor<gradient_form_index> accessor;
        return Coboundary<gradient_form_index, scalar_component_index>::
                value(detail::IdentityStencilEvaluator {},
                      chain,
                      lower_chain,
                      elem,
                      accessor.template access_element<DerivativeIndex>());
    }

    template <
            class ReducedDerivativeIndex,
            class ReconstructedDerivativeIndex,
            class Elem,
            class PositionType>
    [[nodiscard]] KOKKOS_FUNCTION static auto reconstructed_value_(Elem elem, PositionType position)
    {
        using ReconstructionType = Reconstruction<gradient_index_seq, PositionType, Elem>;
        using ReconstructionAccessor
                = tensor::tensor_accessor_for_domain_t<reconstruction_domain_t<gradient_index_seq>>;
        using ReconstructionNaturalElem =
                typename ReconstructionAccessor::natural_domain_t::discrete_element_type;

        auto stencil = reduced_derivative_stencil_<ReducedDerivativeIndex>(elem);
        constexpr std::size_t reduced_derivative_id
                = ddc::type_seq_rank_v<ReducedDerivativeIndex, spatial_index_seq>;
        constexpr std::size_t reconstructed_derivative_id
                = ddc::type_seq_rank_v<ReconstructedDerivativeIndex, spatial_index_seq>;
        auto const reconstruction_natural_elem = detail::natural_elem_from_flat_ids<
                ReconstructionNaturalElem>(
                std::array<std::size_t, 2> {reduced_derivative_id, reconstructed_derivative_id});
        double const reconstruction_coefficient
                = ReconstructionType::value(position, elem, reconstruction_natural_elem);
        stencil *= reconstruction_coefficient;
        return stencil;
    }
};

} // namespace sil::exterior
