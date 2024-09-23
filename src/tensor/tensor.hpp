// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

// struct representing an index mu or nu in a tensor Tmunu.
template <class... CDim>
struct TensorNaturalIndex
{
    using type_seq_dimensions = ddc::detail::TypeSeq<CDim...>;

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

// Helpers to compute the strides of a right layout
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
template <TensorNaturalIndexConcept... TensorNaturalIndex>
struct FullTensorIndex
{
    using type_seq_natural_tensor_indexes = ddc::detail::TypeSeq<TensorNaturalIndex...>;

    static constexpr std::size_t dim_size()
    {
        return (TensorNaturalIndex::dim_size() * ...);
    }

    template <class... CDim>
    static constexpr std::size_t id()
    {
        static_assert(sizeof...(TensorNaturalIndex) == sizeof...(CDim));
        return ((detail::stride<TensorNaturalIndex, TensorNaturalIndex...>()
                 * TensorNaturalIndex::template id<ddc::type_seq_element_t<
                         ddc::type_seq_rank_v<
                                 TensorNaturalIndex,
                                 ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                         ddc::detail::TypeSeq<CDim...>>>())
                + ...);
    }
};

// TensorHandler class, allows to build a domain which represents the tensor and access elements.
template <class... Index>
class TensorHandler
{
private:
    ddc::DiscreteDomain<Index...> const m_tensor_dom;

public:
    explicit TensorHandler();

    ddc::DiscreteDomain<Index...> get_domain();

    template <class... CDim>
    requires(!TensorNaturalIndexConcept<Index> && ...) ddc::DiscreteElement<Index...> get_element();

    template <class... CDim>
    requires(TensorNaturalIndexConcept<Index>&&...) ddc::DiscreteElement<Index...> get_element();
};

template <class... Index>
TensorHandler<Index...>::TensorHandler()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::dim_size())...))
{
}

template <class... Index>
ddc::DiscreteDomain<Index...> TensorHandler<Index...>::get_domain()
{
    return m_tensor_dom;
}

template <class... Index>
template <class... CDim>
requires(!TensorNaturalIndexConcept<Index> && ...)
        ddc::DiscreteElement<Index...> TensorHandler<Index...>::get_element()
{
    return ddc::DiscreteElement<Index...>(
            ddc::DiscreteElement<Index>(Index::template id<CDim...>())...);
}

template <class... Index>
template <class... CDim>
requires(TensorNaturalIndexConcept<Index>&&...)
        ddc::DiscreteElement<Index...> TensorHandler<Index...>::get_element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            Index::template id<ddc::type_seq_element_t<
                    ddc::type_seq_rank_v<Index, ddc::detail::TypeSeq<Index...>>,
                    ddc::detail::TypeSeq<CDim...>>>())...);
}
} // namespace tensor

} // namespace sil
