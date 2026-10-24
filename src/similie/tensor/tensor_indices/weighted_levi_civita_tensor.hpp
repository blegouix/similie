// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <similie/misc/type_seq_ext.hpp>
#include <similie/tensor/tensor_indices/antisymmetric_tensor.hpp>

namespace sil::tensor {

// Source and Target group the natural indices on each side of the Hodge map.
// Store signed coefficients for sorted source and complementary target IDs.
// Permutations within either group change sign; repeated IDs give zero.
template <
        misc::Specialization<TensorAntisymmetricIndex> Source,
        misc::Specialization<TensorAntisymmetricIndex> Target>
struct TensorWeightedLeviCivitaIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_explicitely_stored_tensor = true;
    using subindices_domain_t = misc::cartesian_prod_t<
            typename Source::subindices_domain_t,
            typename Target::subindices_domain_t>;

    KOKKOS_FUNCTION static constexpr subindices_domain_t subindices_domain()
    {
        return subindices_domain_t(Source::subindices_domain(), Target::subindices_domain());
    }
    static constexpr std::size_t rank()
    {
        return Source::rank() + Target::rank();
    }
    static_assert(rank() > 0);
    static_assert([]<class... Indices>(ddc::DiscreteDomain<Indices...> const&) {
        return ((Indices::size() == rank()) && ...);
    }(subindices_domain_t()));
    static constexpr std::size_t size()
    {
        return Source::size() * Target::size();
    }
    static constexpr std::size_t mem_size()
    {
        return Source::mem_size();
    }
    static constexpr std::size_t access_size()
    {
        return 1 + 2 * mem_size();
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(std::array<std::size_t, rank()> const& ids)
    {
        std::array<std::size_t, Source::rank()> source {};
        for (std::size_t i = 0; i < source.size(); ++i)
            source[i] = ids[i];
        return Source::mem_id(source);
    }
    KOKKOS_FUNCTION static constexpr std::size_t access_id(
            std::array<std::size_t, rank()> const& ids)
    {
        bool odd = false;
        for (std::size_t i = 0; i < rank(); ++i) {
            if (ids[i] >= rank())
                return 0;
            for (std::size_t j = i + 1; j < rank(); ++j) {
                if (ids[i] == ids[j])
                    return 0;
                if ((i < Source::rank()) == (j < Source::rank()) && ids[i] > ids[j])
                    odd = !odd;
            }
        }
        return 1 + mem_id(ids) + (odd ? mem_size() : 0);
    }
    KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(std::size_t id)
    {
        assert(id != 0);
        return (id - 1) % mem_size();
    }
    template <class Tensor, class Elem, class Id, class Functor>
    KOKKOS_FUNCTION static constexpr typename Tensor::element_type process_access(
            Functor const& access,
            Tensor const& tensor,
            Elem const& elem)
    {
        std::size_t const id = elem.template uid<Id>();
        if (id == 0)
            return 0.;
        return id > mem_size() ? -access(tensor, elem) : access(tensor, elem);
    }
    KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()>
    mem_id_to_canonical_natural_ids(std::size_t id)
    {
        assert(id < mem_size());
        std::array<std::size_t, rank()> result {};
        std::array const source = Source::mem_id_to_canonical_natural_ids(id);
        for (std::size_t i = 0; i < source.size(); ++i)
            result[i] = source[i];
        std::size_t next = source.size();
        for (std::size_t candidate = 0; candidate < rank(); ++candidate) {
            bool present = false;
            for (std::size_t value : source)
                present = present || candidate == value;
            if (!present)
                result[next++] = candidate;
        }
        return result;
    }
};

static_assert(TensorIndexConcept<TensorWeightedLeviCivitaIndex<
                      TensorAntisymmetricIndex<TensorNaturalIndex<detail::DimensionConcept>>,
                      TensorAntisymmetricIndex<>>>);
} // namespace sil::tensor
