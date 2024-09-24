// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include <boost/math/special_functions/binomial.hpp>

namespace sil {

namespace tensor {

// struct representing an index mu or nu in a tensor Tmunu.
template <class... CDim>
struct TensorNaturalIndex
{
    using type_seq_subindexes = ddc::detail::TypeSeq<>;

    using type_seq_dimensions = ddc::detail::TypeSeq<CDim...>;

    static constexpr std::size_t rank()
    {
        return 1;
    }

    static constexpr std::size_t dim_size()
    {
        return sizeof...(CDim);
    }

    template <class ODim>
    static constexpr std::size_t id()
    {
        return ddc::type_seq_rank_v<ODim, type_seq_dimensions>;
    }
};

// Check that Index derives from TensorNaturalIndex despite the templating
template <class Index>
concept TensorNaturalIndexConcept = requires(Index index)
{
    []<class... T>(TensorNaturalIndex<T...>&) {}(index);
};

// Helpers to compute the strides of a right layout. This is necessary to support non-squared full tensors.
namespace detail
{
    template <std::size_t max_rank, class OTensorNaturalIndex, class... TensorNaturalIndex>
    static constexpr std::size_t stride_factor()
    {
        if constexpr (
                ddc::type_seq_rank_v < OTensorNaturalIndex,
                ddc::detail::TypeSeq < TensorNaturalIndex... >>> max_rank) {
            return OTensorNaturalIndex::dim_size();
        } else {
            return 1;
        }
    }

    template <class OTensorNaturalIndex, class... TensorNaturalIndex>
    static constexpr std::size_t stride()
    {
        return (stride_factor<
                        ddc::type_seq_rank_v<
                                OTensorNaturalIndex,
                                ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                        TensorNaturalIndex,
                        TensorNaturalIndex...>()
                * ...);
    }

} // namespace detail

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for a full tensor (dense with no particular structure).
// template <TensorNaturalIndexConcept... TensorNaturalIndex>
template <class... TensorNaturalIndex>
struct FullTensorIndex
{
    using type_seq_subindexes = ddc::detail::TypeSeq<TensorNaturalIndex...>;

    static constexpr std::size_t rank()
    {
        return (TensorNaturalIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        return (TensorNaturalIndex::dim_size() * ...);
    }

    template <class... CDim>
    static constexpr std::size_t id()
    {
        static_assert(rank() == sizeof...(CDim));
        return ((detail::stride<TensorNaturalIndex, TensorNaturalIndex...>()
                 * TensorNaturalIndex::template id<ddc::type_seq_element_t<
                         ddc::type_seq_rank_v<
                                 TensorNaturalIndex,
                                 ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                         ddc::detail::TypeSeq<CDim...>>>())
                + ...);
    }
};

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for a summetric tensor.
// template <TensorNaturalIndexConcept... TensorNaturalIndex>
template <class... TensorNaturalIndex>
struct SymmetricTensorIndex
{
    using type_seq_subindexes = ddc::detail::TypeSeq<TensorNaturalIndex...>;

    static constexpr std::size_t rank()
    {
        return (TensorNaturalIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        // TODO FIX
        return (TensorNaturalIndex::dim_size() * ...);
    }

    template <class... CDim>
    static constexpr std::size_t id()
    {
        static_assert(rank() == sizeof...(CDim));
        std::array<int, sizeof...(TensorNaturalIndex)> sorted_ids {
                TensorNaturalIndex::template id<ddc::type_seq_element_t<
                        ddc::type_seq_rank_v<
                                TensorNaturalIndex,
                                ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                        ddc::detail::TypeSeq<CDim...>>>()...};
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return boost::math::binomial_coefficient<double>(
                       std::min({TensorNaturalIndex::dim_size()...}) + sizeof...(TensorNaturalIndex)
                               - 1,
                       sizeof...(TensorNaturalIndex))
               - ((sorted_ids[ddc::type_seq_rank_v<
                           TensorNaturalIndex,
                           ddc::detail::TypeSeq<TensorNaturalIndex...>>]
                                   == TensorNaturalIndex::dim_size() - 1
                           ? 0
                           : boost::math::binomial_coefficient<double>(
                                   TensorNaturalIndex::dim_size()
                                           - sorted_ids[ddc::type_seq_rank_v<
                                                   TensorNaturalIndex,
                                                   ddc::detail::TypeSeq<TensorNaturalIndex...>>]
                                           + sizeof...(TensorNaturalIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorNaturalIndex,
                                                   ddc::detail::TypeSeq<TensorNaturalIndex...>> - 2,
                                   sizeof...(TensorNaturalIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorNaturalIndex,
                                                   ddc::detail::TypeSeq<TensorNaturalIndex...>>))
                  + ...)
               - 1;
    }
};

// TensorAccessor class, allows to build a domain which represents the tensor and access elements.
template <class... Index>
class TensorAccessor
{
private:
    ddc::DiscreteDomain<Index...> const m_tensor_dom;

public:
    explicit TensorAccessor();

    ddc::DiscreteDomain<Index...> get_domain();

    template <class... CDim>
    requires(!TensorNaturalIndexConcept<Index> && ...) ddc::DiscreteElement<Index...> get_element();

    template <class... CDim>
    requires(TensorNaturalIndexConcept<Index>&&...) ddc::DiscreteElement<Index...> get_element();
};

template <class... Index>
TensorAccessor<Index...>::TensorAccessor()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::dim_size())...))
{
}

template <class... Index>
ddc::DiscreteDomain<Index...> TensorAccessor<Index...>::get_domain()
{
    return m_tensor_dom;
}

template <class... Index>
template <class... CDim>
requires(!TensorNaturalIndexConcept<Index> && ...)
        ddc::DiscreteElement<Index...> TensorAccessor<Index...>::get_element()
{
    return ddc::DiscreteElement<Index...>(
            ddc::DiscreteElement<Index>(Index::template id<CDim...>())...);
}

namespace detail
{
    template <class Index, class IndexesTypeSeq>
    struct RankBeforeIndex;

    template <class Index, class IndexesTypeSeq>
    struct RankBeforeIndex<Index, ddc::detail::TypeSeq<HeadIndex, Index_...>>
    {
        static constexpr run(std::size_t rank_before_index)
        {
            if constexpr (std::is_same_v<HeadIndex, Index>) {
                return rank_before_index;
            } else {
                return RankBeforeIndex<Index, ddc::detail::TypeSeq<Index_...>>::run(
                        rank_before_index + HeadIndex::rank());
            }
        }
    };

    template <class CDimTypeSeq, std::size_t... Rank>
    using type_seq_dims_at_ranks
            = ddc::detail::TypeSeq<ddc::type_seq_element_t<Rank, CDimTypeSeq>...>;

    template <class Index, class CDimTypeSeq>
    struct IdFromTypeSeqDims;

    template <class Index, class CDimTypeSeq>
    struct IdFromTypeSeqDims<Index, ddc::detail::TypeSeq<CDim...>>
    {
        static constexpr std::size_t run()
        {
            return Index::template id<CDim...>();
        }
    };

    template <class CDim>
    static constexpr std::size_t id_from_type_seq_dim()
    {
        static_assert(rank() == sizeof...(CDim));
        return ((detail::stride<TensorNaturalIndex, TensorNaturalIndex...>()
                 * TensorNaturalIndex::template id<ddc::type_seq_element_t<
                         ddc::type_seq_rank_v<
                                 TensorNaturalIndex,
                                 ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                         ddc::detail::TypeSeq<CDim...>>>())
                + ...);
    }

    template <class Index, class IndexesTypeSeq, class CDim...>
    static constexpr std::size_t recursive_id()
    {
        std::size_t rank_before_index = RankBeforeIndex<Index, IndexesTypeSeq>::run(0);
        return IdFromTypeSeqDims<
                Index,
                typename type_seq_dims_at_ranks<
                        ddc::detail::TypeSeq<CDim...>,
                        rank_before_index
                                + std::make_index_sequence<std::size_t, Index::rank()>>>::run();
    }

} // namespace detail

template <class... Index>
template <class... CDim>
requires(TensorNaturalIndexConcept<Index>&&...)
        ddc::DiscreteElement<Index...> TensorAccessor<Index...>::get_element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            detail::recursive_id<Index, ddc::detail::TypeSeq<Index...>, CDim...>())...);
}
} // namespace tensor

} // namespace sil
