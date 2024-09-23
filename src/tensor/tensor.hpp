// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

// struct representing an index mu in a tensor Tmunu.
template <class... CDim>
struct TensorNaturalIndex
{
    using type_seq_dimensions = ddc::detail::TypeSeq<CDim...>;

    static constexpr std::size_t dim_size()
    {
        return sizeof...(CDim);
    }

    template <class ODim>
    static constexpr std::size_t get_index()
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

// helpers to compute a layout stride
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

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, taking in account the structure (full, symmetric...) of the tensor.
template <TensorNaturalIndexConcept... TensorNaturalIndex>
struct TensorIndex
{
    using type_seq_natural_tensor_indexes = ddc::detail::TypeSeq<TensorNaturalIndex...>;

    static constexpr std::size_t dim_size()
    {
        return (TensorNaturalIndex::dim_size() * ...);
    }

    template <class... CDim>
    static constexpr std::size_t get_index()
    {
        // static_assert same size CDim and TensorNaturalIndex
        // Stride = sum_n (prod_(j = 0 -> n-1) Size(j))*NaturalIndex(n)
        return ((detail::stride<TensorNaturalIndex, TensorNaturalIndex...>()
                 * TensorNaturalIndex::template get_index<ddc::type_seq_element_t<
                         ddc::type_seq_rank_v<
                                 TensorNaturalIndex,
                                 ddc::detail::TypeSeq<TensorNaturalIndex...>>,
                         ddc::detail::TypeSeq<CDim...>>>())
                + ...);
    }
};

// Check that Index derives from TensorIndex despite the templating
template <class Index>
concept TensorIndexConcept = requires(Index index)
{
    []<class T>(TensorIndex<T>&) {}(index);
};

/*
// TensorHandler defined from natural indexes
template <TensorNaturalIndexConcept... Index>
class TensorHandler
{
private:
    ddc::DiscreteDomain<Index...> const m_tensor_dom;

public:
    explicit TensorHandler();

    ddc::DiscreteDomain<Index...> get_domain();

    template <class... CDim>
    ddc::DiscreteElement<Index...> get_element();
};

template <TensorNaturalIndexConcept... Index>
TensorHandler<Index...>::TensorHandler()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::dim_size())...))
{
}

template <TensorNaturalIndexConcept... Index>
ddc::DiscreteDomain<Index...> TensorHandler<Index...>::get_domain()
{
    return m_tensor_dom;
}

template <TensorNaturalIndexConcept... Index>
template <class... CDim>
ddc::DiscreteElement<Index...> TensorHandler<Index...>::get_element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            ddc::type_seq_rank_v<CDim, typename Index::type_seq_dimensions>)...);
}
*/

// TensorHandler defined from unique index
// template <TensorIndexConcept Index> TODO
template <class Index>
class TensorHandler
{
private:
    ddc::DiscreteDomain<Index> const m_tensor_dom;

public:
    explicit TensorHandler();

    ddc::DiscreteDomain<Index> get_domain();

    template <class... CDim>
    ddc::DiscreteElement<Index> get_element();
};

template <TensorIndexConcept Index>
TensorHandler<Index>::TensorHandler()
    : m_tensor_dom(
            ddc::DiscreteElement<Index>(ddc::DiscreteElement<Index>(0)),
            ddc::DiscreteVector<Index>(ddc::DiscreteVector<Index>(Index::dim_size())))
{
}

template <TensorIndexConcept Index>
ddc::DiscreteDomain<Index> TensorHandler<Index>::get_domain()
{
    return m_tensor_dom;
}

template <TensorIndexConcept Index>
template <class... CDim>
ddc::DiscreteElement<Index> TensorHandler<Index>::get_element()
{
    return ddc::DiscreteElement<Index>(Index::template get_index<CDim...>());
}
} // namespace tensor

} // namespace sil
