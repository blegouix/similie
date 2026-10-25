// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <ddc/ddc.hpp>

#include <similie/tensor/tensor_impl.hpp>

namespace sil {

namespace tensor {

/** Store a scalar multiple of the identity using one coefficient.
 * For a matrix, this represents \f$A_{ij} = \alpha\delta_{ij}\f$: every
 * diagonal entry equals the stored value, and every off-diagonal entry is zero.
 * All index dimensions must have the same size. For higher ranks, an entry
 * equals the stored value when all its natural indices agree, and zero otherwise.
 * \important This documentation is fully AI-generated.
 */
template <TensorNatIndex... TensorIndex>
struct TensorScalingIndex
{
    static_assert(
            ((TensorIndex::size()
              == ddc::type_seq_element_t<0, ddc::TypeSeq<TensorIndex...>>::size())
             && ...));

    static constexpr bool is_tensor_index = true;
    static constexpr bool is_explicitely_stored_tensor = true;

    using subindices_domain_t = ddc::DiscreteDomain<TensorIndex...>;

    KOKKOS_FUNCTION static constexpr subindices_domain_t subindices_domain()
    {
        return ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...));
    }

    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t size()
    {
        return (TensorIndex::size() * ...);
    }

    static constexpr std::size_t mem_size()
    {
        return 1;
    }

    static constexpr std::size_t access_size()
    {
        return 1 + mem_size();
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const& natural_ids)
    {
        assert(std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
            return id == *natural_ids.begin();
        }));
        return 0;
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, sizeof...(TensorIndex)> const& natural_ids)
    {
        if (!std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
                return id == *natural_ids.begin();
            })) {
            return 0;
        } else {
            return 1 + mem_id(natural_ids);
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        assert(access_id != 0 && "There is no mem_id associated to access_id=0");
        return access_id - 1;
    }

    template <class Tensor, class Elem, class Id, class FunctorType>
    KOKKOS_FUNCTION static constexpr Tensor::element_type process_access(
            const FunctorType& access,
            Tensor const& tensor,
            Elem const& elem)
    {
        if (elem.template uid<Id>() == 0) {
            return 0.;
        } else {
            return access(tensor, elem);
        }
    }

    KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()>
    mem_id_to_canonical_natural_ids(std::size_t mem_id)
    {
        assert(mem_id < mem_size());
        std::array<std::size_t, rank()> ids;
        std::fill(ids.begin(), ids.end(), 0);
        return ids;
    }
};

static_assert(TensorIndexConcept<TensorScalingIndex<TensorNaturalIndex<detail::DimensionConcept>>>);

} // namespace tensor

} // namespace sil
