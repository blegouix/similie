// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include <boost/math/special_functions/binomial.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for an antisummetric tensor.
template <class... TensorIndex>
struct AntisymmetricTensorIndex
{
    static constexpr bool is_natural_tensor_index = false;

    using subindexes_domain_t = ddc::DiscreteDomain<TensorIndex...>;

    static constexpr subindexes_domain_t subindexes_domain()
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
        return boost::math::binomial_coefficient<
                double>(std::min({TensorIndex::mem_size()...}), sizeof...(TensorIndex));
    }

    static constexpr std::size_t access_size()
    {
        return mem_size() + 1;
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id(
            ddc::DiscreteElement<TensorIndex...> elem)
    {
        // static_assert(rank() == sizeof...(CDim));
        std::array<std::size_t, sizeof...(TensorIndex)> sorted_ids {
                TensorIndex::access_id(ddc::DiscreteElement<TensorIndex>(elem))...};
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {},
                std::vector<std::size_t> {static_cast<std::size_t>(
                        boost::math::binomial_coefficient<double>(
                                std::min({TensorIndex::mem_size()...}),
                                sizeof...(TensorIndex))
                        - ((sorted_ids[ddc::type_seq_rank_v<
                                    TensorIndex,
                                    ddc::detail::TypeSeq<TensorIndex...>>]
                                            == TensorIndex::mem_size() - sizeof...(TensorIndex)
                                                       + ddc::type_seq_rank_v<
                                                               TensorIndex,
                                                               ddc::detail::TypeSeq<TensorIndex...>>
                                    ? 0
                                    : boost::math::binomial_coefficient<double>(
                                            TensorIndex::mem_size()
                                                    - sorted_ids[ddc::type_seq_rank_v<
                                                            TensorIndex,
                                                            ddc::detail::TypeSeq<TensorIndex...>>]
                                                    - 1,
                                            sizeof...(TensorIndex)
                                                    - ddc::type_seq_rank_v<
                                                            TensorIndex,
                                                            ddc::detail::TypeSeq<TensorIndex...>>))
                           + ...)
                        - 1)});
    }

private:
    static constexpr bool permutation_parity(std::array<std::size_t, sizeof...(TensorIndex)> ids)
    {
        bool cnt = false;
        for (int i = 0; i < sizeof...(TensorIndex); i++)
            for (int j = i + 1; j < sizeof...(TensorIndex); j++)
                if (ids[i] > ids[j])
                    cnt = !cnt;
        return cnt;
    }

public:
    static constexpr std::size_t access_id(ddc::DiscreteElement<TensorIndex...> elem)
    {
        std::array<std::size_t, sizeof...(TensorIndex)> ids {
                elem.template uid<TensorIndex>()...}; // better with std::initialize_list ?
        if constexpr (std::all_of(ids.begin(), ids.end(), [&](const std::size_t id) {
                          return id == *ids.begin();
                      })) {
            return 0;
        } else if (!permutation_parity(ids)) {
            return 1 + std::get<1>(mem_id(elem))[0];
        } else {
            return access_size() + std::get<1>(mem_id(elem))[0];
        }
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
    {
        assert(access_id != 0 && "There is no mem_id associated to access_id=0");
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {},
                std::vector<std::size_t> {(access_id - 1) % mem_size()});
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        if (elem.template uid<Id>() == 0) {
            return 0.;
        } else if (elem.template uid<Id>() < access_size()) {
            return access(tensor, elem);
        } else {
            return -access(tensor, elem);
        }
    }
};

namespace detail {
template <class... SubIndex>
struct IsTensorIndex<AntisymmetricTensorIndex<SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
