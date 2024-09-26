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
    using index_type = TensorNaturalIndex<>;

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

    template <class ODim>
    static constexpr std::size_t process_id()
    {
        return id<ODim>();
    }
};

// Helpers to build the id() function which computes the ids of subindexes of an index.
namespace detail {
// For Tmunu and index=nu, returns 1
template <class Index, class...>
struct NbDimsBeforeIndex;

template <class Index, class IndexHead, class... IndexTail>
struct NbDimsBeforeIndex<Index, ddc::detail::TypeSeq<IndexHead, IndexTail...>>
{
    static constexpr std::size_t run(std::size_t nb_dims_before_index)
    {
        if constexpr (std::is_same_v<IndexHead, Index>) {
            return nb_dims_before_index;
        } else {
            return NbDimsBeforeIndex<Index, ddc::detail::TypeSeq<IndexTail...>>::run(
                    nb_dims_before_index + IndexHead::rank());
        }
    }
};

// Offset and index sequence
template <std::size_t Offset, class IndexSeq>
struct OffsetIndexSeq;

template <std::size_t Offset, std::size_t... Is>
struct OffsetIndexSeq<Offset, std::integer_sequence<std::size_t, Is...>>
{
    using type = std::integer_sequence<std::size_t, Offset + Is...>;
};

template <std::size_t Offset, class IndexSeq>
using offset_index_seq_t = OffsetIndexSeq<Offset, IndexSeq>::type;

// Returns dimensions from integers (ie. for Tmunu, <1> gives nu)
template <class CDimTypeSeq, class IndexSeq>
struct TypeSeqDimsAtInts;

template <class CDimTypeSeq, std::size_t... Is>
struct TypeSeqDimsAtInts<CDimTypeSeq, std::integer_sequence<std::size_t, Is...>>
{
    using type = ddc::detail::TypeSeq<ddc::type_seq_element_t<Is, CDimTypeSeq>...>;
};

template <class CDimTypeSeq, class IndexSeq>
using type_seq_dims_at_ints_t = TypeSeqDimsAtInts<CDimTypeSeq, IndexSeq>::type;

// Returns Index::id but from a type seq (in place of a variadic template CDim...)
template <class Index, class TypeSeqDims>
struct IdFromTypeSeqDims;

template <class Index, class... CDim>
struct IdFromTypeSeqDims<Index, ddc::detail::TypeSeq<CDim...>>
{
    static constexpr std::size_t run()
    {
        // return Index::template id<CDim...>();
        return Index::template process_id<CDim...>();
    }
};

// Returns Index::id for the subindex Index of the IndexesTypeSeq
template <class Index, class IndexesTypeSeq, class... CDim>
static constexpr std::size_t id()
{
    return IdFromTypeSeqDims<
            Index,
            type_seq_dims_at_ints_t<
                    ddc::detail::TypeSeq<CDim...>,
                    offset_index_seq_t<
                            NbDimsBeforeIndex<Index, IndexesTypeSeq>::run(0),
                            std::make_integer_sequence<std::size_t, Index::rank()>>>>::run();
}

} // namespace detail

// Helpers to compute the strides of a right layout. This is necessary to support non-squared full tensors.
namespace detail {
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
template <class... TensorIndex>
struct FullTensorIndex
{
    using index_type = FullTensorIndex<>;

    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        return (TensorIndex::dim_size() * ...);
    }

    template <class... CDim>
    static constexpr std::size_t id()
    {
        //static_assert(rank() == sizeof...(CDim));
        return ((detail::stride<TensorIndex, TensorIndex...>()
                 * detail::id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>())
                + ...);
    }

    template <class... CDim>
    static constexpr std::size_t process_id()
    {
        return id<CDim...>();
    }
};

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for a summetric tensor.
template <class... TensorIndex>
struct SymmetricTensorIndex
{
    using index_type = SymmetricTensorIndex<>;

    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        return boost::math::binomial_coefficient<double>(
                std::min({TensorIndex::dim_size()...}) + sizeof...(TensorIndex) - 1,
                sizeof...(TensorIndex));
    }

    template <class... CDim>
    static constexpr std::size_t id()
    {
        // static_assert(rank() == sizeof...(CDim));
        std::array<int, sizeof...(TensorIndex)> sorted_ids {
                detail::id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>()...};
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return boost::math::binomial_coefficient<double>(
                       std::min({TensorIndex::dim_size()...}) + sizeof...(TensorIndex) - 1,
                       sizeof...(TensorIndex))
               - ((sorted_ids[ddc::type_seq_rank_v<
                           TensorIndex,
                           ddc::detail::TypeSeq<TensorIndex...>>]
                                   == TensorIndex::dim_size() - 1
                           ? 0
                           : boost::math::binomial_coefficient<double>(
                                   TensorIndex::dim_size()
                                           - sorted_ids[ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>]
                                           + sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>> - 2,
                                   sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>))
                  + ...)
               - 1;
    }

    template <class... CDim>
    static constexpr std::size_t process_id()
    {
        return id<CDim...>();
    }
};

// struct representing an abstract unique index sweeping on all possible combination of natural indexes, for an antisummetric tensor.
template <class... TensorIndex>
struct AntisymmetricTensorIndex
{
    using index_type = AntisymmetricTensorIndex<>;

    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t dim_size()
    {
        return boost::math::binomial_coefficient<double>(
                std::min({TensorIndex::dim_size()...}) + sizeof...(TensorIndex) - 2,
                sizeof...(TensorIndex));
    }

    template <class... CDim>
    static constexpr std::size_t id()
    {
        // static_assert(rank() == sizeof...(CDim));
        std::array<int, sizeof...(TensorIndex)> sorted_ids {
                detail::id<TensorIndex, ddc::detail::TypeSeq<TensorIndex...>, CDim...>()...};
        std::sort(sorted_ids.begin(), sorted_ids.end());
        return boost::math::binomial_coefficient<double>(
                       std::min({TensorIndex::dim_size()...}) + sizeof...(TensorIndex) - 1,
                       sizeof...(TensorIndex))
               - ((sorted_ids[ddc::type_seq_rank_v<
                           TensorIndex,
                           ddc::detail::TypeSeq<TensorIndex...>>]
                                   == TensorIndex::dim_size() - 1
                           ? 0
                           : boost::math::binomial_coefficient<double>(
                                   TensorIndex::dim_size()
                                           - sorted_ids[ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>]
                                           + sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>> - 2,
                                   sizeof...(TensorIndex)
                                           - ddc::type_seq_rank_v<
                                                   TensorIndex,
                                                   ddc::detail::TypeSeq<TensorIndex...>>))
                  + ...)
               - 2;
    }


private:
    template <class Head, class... Tail>
    inline static constexpr bool are_all_same = (std::is_same_v<Head, Tail> && ...);

public:
    template <class... CDim>
    static constexpr std::size_t process_id()
    {
        if constexpr (are_all_same<CDim...>) {
            return 0;
        } else {
            return id<CDim...>() + 1;
        }
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

    ddc::DiscreteDomain<Index...> domain();

    template <class... CDim>
    ddc::DiscreteElement<Index...> element();

    // getter
    template <class T, class Domain, class MemorySpace, class... DDim>
    T operator()(
            ddc::ChunkSpan<T, Domain, std::experimental::layout_right, MemorySpace> tensor_field,
            ddc::DiscreteElement<DDim...> elem);

    // TODO operator[] ?

    // setter
    template <class T, class Domain, class MemorySpace, class... DDim>
    void set(
            ddc::ChunkSpan<T, Domain, std::experimental::layout_right, MemorySpace> tensor_field,
            ddc::DiscreteElement<DDim...> elem,
            T value);
};

template <class... Index>
TensorAccessor<Index...>::TensorAccessor()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::dim_size())...))
{
}

template <class... Index>
ddc::DiscreteDomain<Index...> TensorAccessor<Index...>::domain()
{
    return m_tensor_dom;
}

template <class... Index>
template <class... CDim>
ddc::DiscreteElement<Index...> TensorAccessor<Index...>::element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            detail::id<Index, ddc::detail::TypeSeq<Index...>, CDim...>())...);
}

// Helpers to handle antisymmetry (eventual multiplication with -1) or non-stored zeros
namespace detail {
template <
        class TensorField,
        class Element,
        class IndexHeadsTypeSeq,
        class IndexInterest,
        class... IndexTail>
struct Access;

template <class TensorField, class Element, class... IndexHead, class IndexInterest>
struct Access<TensorField, Element, ddc::detail::TypeSeq<IndexHead...>, IndexInterest>
{
    static constexpr TensorField::element_type run(TensorField tensor_field, Element elem)
    {
        if constexpr (std::is_same_v<
                              typename IndexInterest::index_type,
                              AntisymmetricTensorIndex<>>) {
            std::cout << detail::
                            id<IndexInterest, ddc::detail::TypeSeq<IndexHead..., IndexInterest>>();
            if constexpr (
                    detail::id<IndexInterest, ddc::detail::TypeSeq<IndexHead..., IndexInterest>>()
                    > sizeof...(IndexHead) + 1) {
                return -tensor_field(elem);
            } else if (
                    detail::id<IndexInterest, ddc::detail::TypeSeq<IndexHead..., IndexInterest>>()
                    == sizeof...(IndexHead) + 1) {
                return 0;
            } else {
                return tensor_field(elem);
            }
        } else {
            return tensor_field(elem);
        }
    }
};

template <
        class TensorField,
        class Element,
        class... IndexHead,
        class IndexInterest,
        class... IndexTail>
struct Access<TensorField, Element, ddc::detail::TypeSeq<IndexHead...>, IndexInterest, IndexTail...>
{
    static constexpr TensorField::element_type run(TensorField tensor_field, Element elem)
    {
        if constexpr (std::is_same_v<
                              typename IndexInterest::index_type,
                              AntisymmetricTensorIndex<>>) {
            if constexpr (
                    detail::id<
                            IndexInterest,
                            ddc::detail::TypeSeq<IndexHead..., IndexInterest, IndexTail...>>()
                    > sizeof...(IndexHead) + 1) {
                return -Access<
                        TensorField,
                        Element,
                        ddc::detail::TypeSeq<IndexHead..., IndexInterest>,
                        IndexTail...>::run(tensor_field, elem);
            } else if (
                    detail::id<
                            IndexInterest,
                            ddc::detail::TypeSeq<IndexHead..., IndexInterest, IndexTail...>>()
                    == sizeof...(IndexHead) + 1) {
                return 0;
            } else {
                return Access<
                        TensorField,
                        Element,
                        ddc::detail::TypeSeq<IndexHead..., IndexInterest>,
                        IndexTail...>::run(tensor_field, elem);
            }
        } else {
            return Access<
                    TensorField,
                    Element,
                    ddc::detail::TypeSeq<IndexHead..., IndexInterest>,
                    IndexTail...>::run(tensor_field, elem);
        }
    }
};

} // namespace detail

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class Tensor : public ddc::ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>
{
public:
    using ddc::ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>::ChunkSpan;
    using ddc::ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>::reference;

    template <class... DElems>
    KOKKOS_FUNCTION constexpr typename ddc::
            ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>::reference
            operator()(DElems const&... delems) const noexcept
    {
        return ddc::ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>::
        operator()(delems...);
    }
};

template <class... Index>
template <class T, class Domain, class MemorySpace, class... DDim>
T TensorAccessor<Index...>::operator()(
        ddc::ChunkSpan<T, Domain, std::experimental::layout_right, MemorySpace> tensor_field,
        ddc::DiscreteElement<DDim...> elem)
{
    /*
    return detail::Access<
            ddc::ChunkSpan<T, Domain, std::experimental::layout_right, MemorySpace>,
            ddc::DiscreteElement<DDim...>,
            ddc::detail::TypeSeq<>,
            Index...>::run(tensor_field, elem);
*/
    return tensor_field(elem);
}

template <class... Index>
template <class T, class Domain, class MemorySpace, class... DDim>
void TensorAccessor<Index...>::set(
        ddc::ChunkSpan<T, Domain, std::experimental::layout_right, MemorySpace> tensor_field,
        ddc::DiscreteElement<DDim...> elem,
        T value)
{
    tensor_field(elem) = value;
}

} // namespace tensor

} // namespace sil
