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

    static constexpr std::size_t rank()
    {
        return 1;
    }

    static constexpr std::size_t size()
    {
        return sizeof...(CDim);
    }

    static constexpr std::size_t mem_size()
    {
        return size();
    }

    static constexpr std::size_t access_size()
    {
        return size();
    }

    template <class ODim>
    static constexpr std::size_t mem_id()
    {
        return ddc::type_seq_rank_v<ODim, type_seq_dimensions>;
    }

    template <class ODim>
    static constexpr std::size_t access_id()
    {
        return mem_id<ODim>();
    }

    static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        return access_id;
    }

    template <class Tensor, class Elem>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        return access(tensor, elem);
    }
};

// Helpers to build the access_id() function which computes the ids of subindexes of an index.
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

// Returns Index::access_id but from a type seq (in place of a variadic template CDim...)
template <class Index, class TypeSeqDims>
struct IdFromTypeSeqDims;

template <class Index, class... CDim>
struct IdFromTypeSeqDims<Index, ddc::detail::TypeSeq<CDim...>>
{
    static constexpr std::size_t run()
    {
        return Index::template access_id<CDim...>();
    }
};

// Returns Index::access_id for the subindex Index of the IndexesTypeSeq
template <class Index, class IndexesTypeSeq, class... CDim>
static constexpr std::size_t access_id()
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

// TensorAccessor class, allows to build a domain which represents the tensor and access elements.
template <class... Index>
class TensorAccessor
{
public:
    explicit constexpr TensorAccessor();

    static constexpr ddc::DiscreteDomain<Index...> mem_domain();

    static constexpr ddc::DiscreteDomain<Index...> access_domain();

    template <class... CDim>
    static constexpr ddc::DiscreteElement<Index...> element();
};

template <class... Index>
constexpr TensorAccessor<Index...>::TensorAccessor()
{
}

template <class... Index>
constexpr ddc::DiscreteDomain<Index...> TensorAccessor<Index...>::mem_domain()
{
    return ddc::DiscreteDomain<Index...>(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::mem_size())...));
}

template <class... Index>
constexpr ddc::DiscreteDomain<Index...> TensorAccessor<Index...>::access_domain()
{
    return ddc::DiscreteDomain<Index...>(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::access_size())...));
}

template <class... Index>
template <class... CDim>
constexpr ddc::DiscreteElement<Index...> TensorAccessor<Index...>::element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            detail::access_id<Index, ddc::detail::TypeSeq<Index...>, CDim...>())...);
}

// Helpers to handle memory access and processing for particular tensor structures (ie. eventual multiplication with -1 for antisymmetry or non-stored zeros)
namespace detail {
template <class DDim>
struct IsTensorIndex
{
    using type = std::true_type; // TODO FIX
};

template <class... SubIndex>
struct IsTensorIndex<TensorNaturalIndex<SubIndex...>>
{
    using type = std::true_type;
};

template <class DDim>
static constexpr bool is_tensor_index_v = IsTensorIndex<DDim>::type::value;

template <class DDimInterest, class... DDim>
ddc::DiscreteElement<DDim...> replace_access_id_with_mem_id(ddc::DiscreteElement<DDim...> elem)
{
    return ddc::DiscreteElement<DDim...>(
            (std::is_same_v<DDimInterest, DDim> && detail::is_tensor_index_v<DDim>
                     ? DDim::access_id_to_mem_id(ddc::DiscreteElement<DDim>(elem).uid())
                     : ddc::DiscreteElement<DDim>(elem).uid())...);
}
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
    template <class Elem>
    static TensorField::element_type run(TensorField tensor_field, Elem const& elem)
    {
        if constexpr (detail::is_tensor_index_v<IndexInterest>) {
            return IndexInterest::template process_access<TensorField, Elem>(
                    [](TensorField tensor_field_, Elem elem_) -> TensorField::element_type {
                        return tensor_field_(elem_);
                    },
                    tensor_field,
                    elem);
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
    template <class Elem>
    static TensorField::element_type run(TensorField tensor_field, Elem const& elem)
    {
        if constexpr (detail::is_tensor_index_v<IndexInterest>) {
            return IndexInterest::template process_access<TensorField, Elem>(
                    [](TensorField tensor_field_, Elem elem_) -> TensorField::element_type {
                        return Access<
                                TensorField,
                                Element,
                                ddc::detail::TypeSeq<IndexHead..., IndexInterest>,
                                IndexTail...>::run(tensor_field_, elem_);
                    },
                    tensor_field,
                    elem);
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
class Tensor
{
};

} // namespace tensor

} // namespace sil

namespace ddc {

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool enable_chunk<
        sil::tensor::Tensor<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>> = true;

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool enable_borrowed_chunk<
        sil::tensor::Tensor<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>> = true;

} // namespace ddc

namespace sil {

namespace tensor {

template <class ElementType, class... DDim, class LayoutStridedPolicy, class MemorySpace>
class Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
    : public ddc::
              ChunkSpan<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
{
public:
    using ddc::
            ChunkSpan<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    ChunkSpan;
    using reference = ddc::
            ChunkSpan<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
                    reference;

    using ddc::
            ChunkSpan<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>::
            operator();

    // TODO operator[] ?

    template <class... DElems>
    KOKKOS_FUNCTION ElementType get(DElems const&... delems) const noexcept
    {
        return detail::Access<
                Tensor<ElementType,
                       ddc::DiscreteDomain<DDim...>,
                       std::experimental::layout_right,
                       MemorySpace>,
                ddc::DiscreteElement<DDim...>,
                ddc::detail::TypeSeq<>,
                DDim...>::run(*this, ddc::DiscreteElement(delems...));
    }

    template <class... DElems>
    KOKKOS_FUNCTION constexpr reference operator()(DElems const&... delems) const noexcept
    {
        return ddc::ChunkSpan<
                ElementType,
                ddc::DiscreteDomain<DDim...>,
                LayoutStridedPolicy,
                MemorySpace>::
        operator()(ddc::DiscreteElement<DDim...>(
                (detail::is_tensor_index_v<DDim>
                         ? DDim::access_id_to_mem_id(ddc::DiscreteElement<DDim>(delems...).uid())
                         : ddc::DiscreteElement<DDim>(delems...).uid())...));
    }

    void fill_using_lambda(std::function<
                           void(Tensor<ElementType,
                                       ddc::DiscreteDomain<DDim...>,
                                       LayoutStridedPolicy,
                                       MemorySpace>,
                                ddc::DiscreteElement<DDim...>)> lambda_func)
    {
        ddc::for_each(this->domain(), [&](ddc::DiscreteElement<DDim...> elem) {
            lambda_func(*this, elem);
        });
    }
};

} // namespace tensor

} // namespace sil
