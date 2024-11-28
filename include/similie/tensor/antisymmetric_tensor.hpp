// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/binomial_coefficient.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indices, for an antisymmetric tensor.
template <class... TensorIndex>
struct TensorAntisymmetricIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_explicitely_stored_tensor = true;

    using subindices_domain_t = ddc::DiscreteDomain<TensorIndex...>;

    static constexpr subindices_domain_t subindices_domain()
    {
        return ddc::DiscreteDomain<TensorIndex...>(
                ddc::DiscreteElement<TensorIndex...>(ddc::DiscreteElement<TensorIndex>(0)...),
                ddc::DiscreteVector<TensorIndex...>(
                        ddc::DiscreteVector<TensorIndex>(TensorIndex::size())...));
    }

    static constexpr std::size_t rank()
    {
        if constexpr (sizeof...(TensorIndex) == 0) {
            return 0;
        } else {
            return (TensorIndex::rank() + ...);
        }
    }

    static constexpr std::size_t size()
    {
        if constexpr (rank() == 0) {
            return 1;
        } else {
            return (TensorIndex::size() + ...);
        }
    }

    static constexpr std::size_t mem_size()
    {
        if constexpr (rank() == 0) {
            return 1;
        } else {
            return misc::binomial_coefficient(
                    ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size(),
                    rank());
        }
    }

    static constexpr std::size_t access_size()
    {
        if constexpr (rank() <= 1) {
            return mem_size();
        } else {
            return mem_size() + 1;
        }
    }

    static constexpr std::size_t mem_id(std::array<std::size_t, rank()> const natural_ids)
    {
        std::array<std::size_t, rank()> sorted_ids(natural_ids);
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return mem_size()
               - (0 + ...
                  + (sorted_ids[ddc::type_seq_rank_v<
                             TensorIndex,
                             ddc::detail::TypeSeq<TensorIndex...>>]
                                     == TensorIndex::mem_size() - rank()
                                                + ddc::type_seq_rank_v<
                                                        TensorIndex,
                                                        ddc::detail::TypeSeq<TensorIndex...>>
                             ? 0
                             : misc::binomial_coefficient(
                                       TensorIndex::mem_size()
                                               - sorted_ids[ddc::type_seq_rank_v<
                                                       TensorIndex,
                                                       ddc::detail::TypeSeq<TensorIndex...>>]
                                               - 1,
                                       rank()
                                               - ddc::type_seq_rank_v<
                                                       TensorIndex,
                                                       ddc::detail::TypeSeq<TensorIndex...>>)))
               - 1;
    }

private:
    static constexpr bool permutation_parity(std::array<std::size_t, rank()> ids)
    {
        bool cnt = false;
        for (int i = 0; i < rank(); i++)
            for (int j = i + 1; j < rank(); j++)
                if (ids[i] > ids[j])
                    cnt = !cnt;
        return cnt;
    }

public:
    static constexpr std::size_t access_id(std::array<std::size_t, rank()> const natural_ids)
    {
        if constexpr (rank() <= 1) {
            return mem_id(natural_ids);
        } else {
            if (std::all_of(natural_ids.begin(), natural_ids.end(), [&](const std::size_t id) {
                    return id == *natural_ids.begin();
                })) {
                return 0;
            } else if (!permutation_parity(natural_ids)) {
                return 1 + mem_id(natural_ids);
            } else {
                return access_size() + mem_id(natural_ids);
            }
        }
    }

    static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        if constexpr (rank() <= 1) {
            return access_id;
        } else {
            if (access_id != 0) {
                return (access_id - 1) % mem_size();
            } else {
                return std::numeric_limits<std::size_t>::max();
            }
        }
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        if constexpr (rank() <= 1) {
            return access(tensor, elem);
        } else {
            if (elem.template uid<Id>() == 0) {
                return 0.;
            } else if (elem.template uid<Id>() < access_size()) {
                return access(tensor, elem);
            } else {
                return -access(tensor, elem);
            }
        }
    }

    static constexpr std::array<std::size_t, rank()> mem_id_to_canonical_natural_ids(
            std::size_t mem_id)
    {
        assert(mem_id < mem_size());
        std::array<std::size_t, rank()> ids;
        std::size_t d
                = ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::mem_size();
        std::size_t r = rank();
        for (std::size_t i = 0; i < rank(); ++i) {
            const std::size_t triangle_size = misc::binomial_coefficient(d, r - i);
            for (std::size_t j = 0; j < d; ++j) {
                const std::size_t subtriangle_size = misc::binomial_coefficient(d - j - 1, r - i);
                if (triangle_size - subtriangle_size > mem_id) {
                    ids[i] = ddc::type_seq_element_t<0, ddc::detail::TypeSeq<TensorIndex...>>::
                                     mem_size()
                             - d + j;
                    mem_id -= triangle_size - misc::binomial_coefficient(d - j, r - i);
                    d -= j + 1;
                    break;
                }
                ids[i] = 0;
            }
        }
        return ids;
    }
};

namespace detail {

template <class T>
struct ToTensorAntisymmetricIndex;

template <tensor::TensorNatIndex Index>
    requires(Index::rank() == 0)
struct ToTensorAntisymmetricIndex<Index>
{
    using type = TensorAntisymmetricIndex<>;
};

template <tensor::TensorNatIndex Index>
    requires(Index::rank() > 0)
struct ToTensorAntisymmetricIndex<Index>
{
    using type = TensorAntisymmetricIndex<Index>;
};

template <tensor::TensorNatIndex... Index>
struct ToTensorAntisymmetricIndex<TensorAntisymmetricIndex<Index...>>
{
    using type = TensorAntisymmetricIndex<Index...>;
};

} // namespace detail

template <class T>
using to_tensor_antisymmetric_index_t = typename detail::ToTensorAntisymmetricIndex<T>::type;

} // namespace tensor

} // namespace sil
