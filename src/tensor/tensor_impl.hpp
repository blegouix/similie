// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

// struct representing an index mu or nu in a tensor Tmunu.
template <class... CDim>
struct TensorNaturalIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_tensor_natural_index = true;

    using type_seq_dimensions = ddc::detail::TypeSeq<CDim...>;

    using subindices_domain_t = ddc::DiscreteDomain<>;

    static constexpr subindices_domain_t subindices_domain()
    {
        return ddc::DiscreteDomain<>();
    }

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
    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id()
    {
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {1.},
                std::vector<std::size_t> {ddc::type_seq_rank_v<ODim, type_seq_dimensions>});
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id(
            std::size_t const id)
    {
        return std::pair<
                std::vector<double>,
                std::vector<std::size_t>>(std::vector<double> {1.}, std::vector<std::size_t> {id});
    }

    static constexpr std::size_t access_id(std::size_t const id)
    {
        return id;
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
    {
        return std::pair<std::vector<double>, std::vector<std::size_t>>(
                std::vector<double> {1.},
                std::vector<std::size_t> {access_id});
    }

    template <class Tensor, class Elem, class Id>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        return access(tensor, elem);
    }
};

template <class DDim>
concept TensorIndex = requires {
    { DDim::is_tensor_index } -> std::convertible_to<bool>;
} && DDim::is_tensor_index;

template <class DDim>
concept TensorNatIndex = requires {
    { DDim::is_tensor_natural_index } -> std::convertible_to<bool>;
} && DDim::is_tensor_natural_index;

// Helpers to build the access_id() function which computes the ids of subindices of an index. This cumbersome logic is necessary because subindices do not necessarily have the same rank.
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
template <class Index, class SubindicesDomain, class TypeSeqDims>
struct IdFromTypeSeqDims;

template <class Index, class... Subindex, class... CDim>
struct IdFromTypeSeqDims<Index, ddc::DiscreteDomain<Subindex...>, ddc::detail::TypeSeq<CDim...>>
{
    static constexpr std::size_t run()
    {
        static_assert(sizeof...(Subindex) == sizeof...(CDim));
        if constexpr (TensorNatIndex<Index>) {
            return Index::access_id(
                    ddc::type_seq_rank_v<CDim, typename Index::type_seq_dimensions>...);
        } else {
            return Index::access_id(std::array<
                                    std::size_t,
                                    sizeof...(Subindex)> {ddc::type_seq_rank_v<
                    typename ddc::type_seq_element_t<
                            ddc::type_seq_rank_v<Subindex, ddc::detail::TypeSeq<Subindex...>>,
                            ddc::detail::TypeSeq<CDim...>>,
                    typename Subindex::type_seq_dimensions>...});
        }
    }
};

// Returns Index::access_id for the subindex Index of the IndicesTypeSeq
template <class Index, class IndicesTypeSeq, class... CDim>
static constexpr std::size_t access_id() // TODO consteval. This is not compile-time atm :/
{
    if constexpr (TensorNatIndex<Index>) {
        return IdFromTypeSeqDims<
                Index,
                ddc::DiscreteDomain<Index>,
                type_seq_dims_at_ints_t<
                        ddc::detail::TypeSeq<CDim...>,
                        offset_index_seq_t<
                                NbDimsBeforeIndex<Index, IndicesTypeSeq>::run(0),
                                std::make_integer_sequence<std::size_t, Index::rank()>>>>::run();
    } else {
        return IdFromTypeSeqDims<
                Index,
                typename Index::subindices_domain_t,
                type_seq_dims_at_ints_t<
                        ddc::detail::TypeSeq<CDim...>,
                        offset_index_seq_t<
                                NbDimsBeforeIndex<Index, IndicesTypeSeq>::run(0),
                                std::make_integer_sequence<std::size_t, Index::rank()>>>>::run();
    }
}

template <class Index, class SubindicesDomain>
struct IdFromElem;

template <class Index, class... Subindex>
struct IdFromElem<Index, ddc::DiscreteDomain<Subindex...>>
{
    template <class Elem>
    static constexpr std::size_t run(Elem natural_elem)
    {
        if constexpr (TensorNatIndex<Index>) {
            return Index::access_id(natural_elem.template uid<Index>());
        } else {
            return Index::access_id(std::array<std::size_t, sizeof...(Subindex)> {
                    natural_elem.template uid<Subindex>()...});
        }
    }
};

template <class Index, class IndicesTypeSeq, class... NaturalIndex>
static constexpr std::size_t access_id(
        ddc::DiscreteElement<NaturalIndex...>
                natural_elem) // TODO consteval. This is not compile-time atm :/
{
    if constexpr (TensorNatIndex<Index>) {
        return IdFromElem<Index, ddc::DiscreteDomain<Index>>::run(natural_elem);
    } else {
        return IdFromElem<Index, typename Index::subindices_domain_t>::run(natural_elem);
    }
}

} // namespace detail

// TensorAccessor class, allows to build a domain which represents the tensor and access elements.
template <class... Index>
class TensorAccessor
{
public:
    explicit constexpr TensorAccessor();

    using natural_domain_t = ddc::cartesian_prod_t<std::conditional_t<
            TensorNatIndex<Index>,
            ddc::DiscreteDomain<Index>,
            typename Index::subindices_domain_t>...>;

    static constexpr natural_domain_t natural_domain();

    static constexpr ddc::DiscreteDomain<Index...> mem_domain();

    static constexpr ddc::DiscreteDomain<Index...> access_domain();

    template <class... CDim>
    static constexpr ddc::DiscreteElement<Index...> access_element();

    template <class... NaturalIndex>
    static constexpr ddc::DiscreteElement<Index...> access_element(
            ddc::DiscreteElement<NaturalIndex...> natural_elem);
};

namespace detail {

template <class Dom>
struct TensorAccessorForDomain;

template <class... Index>
struct TensorAccessorForDomain<ddc::DiscreteDomain<Index...>>
{
    using type = TensorAccessor<Index...>;
};

} // namespace detail

template <class Dom>
using tensor_accessor_for_domain_t = detail::TensorAccessorForDomain<Dom>::type;

template <class... Index>
constexpr TensorAccessor<Index...>::TensorAccessor()
{
}

namespace detail {
template <class Index>
constexpr auto natural_domain()
{
    if constexpr (TensorNatIndex<Index>) {
        return typename ddc::DiscreteDomain<
                Index>(ddc::DiscreteElement<Index>(0), ddc::DiscreteVector<Index>(Index::size()));
    } else {
        return Index::subindices_domain();
    }
}
} // namespace detail

template <class... Index>
constexpr TensorAccessor<Index...>::natural_domain_t TensorAccessor<Index...>::natural_domain()
{
    return natural_domain_t(detail::natural_domain<Index>()...);
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
constexpr ddc::DiscreteElement<Index...> TensorAccessor<Index...>::access_element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            detail::access_id<Index, ddc::detail::TypeSeq<Index...>, CDim...>())...);
}

template <class... Index>
template <class... NaturalIndex>
constexpr ddc::DiscreteElement<Index...> TensorAccessor<Index...>::access_element(
        ddc::DiscreteElement<NaturalIndex...> natural_elem)
{
    return ddc::DiscreteElement<Index...>(
            ddc::DiscreteElement<Index>(detail::access_id<Index, ddc::detail::TypeSeq<Index...>>(
                    typename natural_domain_t::discrete_element_type(natural_elem)))...);
}

namespace detail {

// Helpers to handle memory access and processing for particular tensor structures (ie. eventual multiplication with -1 for antisymmetry or non-stored zeros)
template <
        class TensorField,
        class Element,
        class IndexHeadsTypeSeq,
        class IndexInterest,
        class... IndexTail>
struct Access;

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
        /*
         ----- Important warning -----
         The general case is not correctly handled here. It would be difficult to do so.
         It means you can get silent bug (with wrong result) if you try to use exotic ordering
         of dimensions/indices. Ie., a TensorYoungTableauIndex has to be the last of the list.
         */
        if constexpr (sizeof...(IndexTail) > 0) {
            if constexpr (TensorIndex<IndexInterest>) {
                return IndexInterest::template process_access<TensorField, Elem, IndexInterest>(
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
        } else {
            if constexpr (TensorIndex<IndexInterest>) {
                return IndexInterest::template process_access<TensorField, Elem, IndexInterest>(
                        [](TensorField tensor_field_, Elem elem_) -> TensorField::element_type {
                            std::pair<std::vector<double>, std::vector<std::size_t>> mem_id
                                    = IndexInterest::access_id_to_mem_id(
                                            elem_.template uid<IndexInterest>());

                            double tensor_field_value = 0;
                            if (std::get<0>(mem_id).size() > 0) {
                                for (std::size_t i = 0; i < std::get<0>(mem_id).size(); ++i) {
                                    tensor_field_value
                                            += std::get<0>(mem_id)[i]
                                               * tensor_field_
                                                         .mem(ddc::DiscreteElement<IndexHead...>(
                                                                      elem_),
                                                              ddc::DiscreteElement<IndexInterest>(
                                                                      std::get<1>(mem_id)[i]));
                                }
                            } else {
                                tensor_field_value = 1.;
                            }

                            return tensor_field_value;
                        },
                        tensor_field,
                        elem);
            } else {
                return tensor_field(elem);
            }
        }
    }
};

// Functor for memory element access (if defined)
template <typename InterestDim>
struct LambdaMemElem
{
    static ddc::DiscreteElement<InterestDim> run(ddc::DiscreteElement<InterestDim> elem)
    {
        return elem;
    }
};

template <TensorIndex InterestDim>
struct LambdaMemElem<InterestDim>
{
    static ddc::DiscreteElement<InterestDim> run(ddc::DiscreteElement<InterestDim> elem)
    {
        // TODO static_assert unique mem_id
        return ddc::DiscreteElement<InterestDim>(std::get<1>(InterestDim::access_id_to_mem_id(
                ddc::DiscreteElement<InterestDim>(elem).uid()))[0]);
    }
};

} // namespace detail

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class Tensor;

} // namespace tensor

} // namespace sil

// @cond

namespace ddc {

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool enable_chunk<
        sil::tensor::Tensor<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool enable_borrowed_chunk<
        sil::tensor::Tensor<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

} // namespace ddc

// @endcond

namespace sil {

namespace tensor {

/// Tensor class
template <class ElementType, class... DDim, class LayoutStridedPolicy, class MemorySpace>
class Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
    : public ddc::
              ChunkSpan<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
{
protected:
    using base_type = ddc::
            ChunkSpan<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>;

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

    KOKKOS_FUNCTION constexpr explicit Tensor(ddc::ChunkSpan<
                                              ElementType,
                                              ddc::DiscreteDomain<DDim...>,
                                              LayoutStridedPolicy,
                                              MemorySpace> other) noexcept
        : base_type(other)
    {
    }

    using accessor_t = TensorAccessor<DDim...>;

    static constexpr accessor_t accessor()
    {
        return accessor_t();
    }

    using natural_domain_t = accessor_t::natural_domain_t;

    static constexpr natural_domain_t natural_domain()
    {
        return accessor_t::natural_domain();
    }

    static constexpr ddc::DiscreteDomain<DDim...> mem_domain()
    {
        return accessor_t::mem_domain();
    }

    static constexpr ddc::DiscreteDomain<DDim...> access_domain()
    {
        return accessor_t::access_domain();
    }

    template <class... CDim>
    static constexpr ddc::DiscreteElement<DDim...> access_element()
    {
        return accessor_t::template access_element<CDim...>();
    }

    template <class... NaturalIndex>
    static constexpr ddc::DiscreteElement<DDim...> access_element(
            ddc::DiscreteElement<NaturalIndex...> natural_elem)
    {
        return accessor_t::access_element(natural_elem);
    }

    template <class... DElems>
    KOKKOS_FUNCTION ElementType get(DElems const&... delems) const noexcept
    {
        return detail::Access<
                Tensor<ElementType,
                       ddc::DiscreteDomain<DDim...>,
                       Kokkos::layout_right,
                       MemorySpace>,
                ddc::DiscreteElement<DDim...>,
                ddc::detail::TypeSeq<>,
                DDim...>::run(*this, ddc::DiscreteElement(delems...));
    }

    template <class... DElems>
    KOKKOS_FUNCTION constexpr reference mem(DElems const&... delems) const noexcept
    {
        return ddc::ChunkSpan<
                ElementType,
                ddc::DiscreteDomain<DDim...>,
                LayoutStridedPolicy,
                MemorySpace>::operator()(delems...);
    }

    template <class... DElems>
    KOKKOS_FUNCTION constexpr reference operator()(DElems const&... delems) const noexcept
    {
        // TODO static_assert unique mem_id
        return ddc::ChunkSpan<
                ElementType,
                ddc::DiscreteDomain<DDim...>,
                LayoutStridedPolicy,
                MemorySpace>::
        operator()(ddc::DiscreteElement<DDim...>(
                detail::LambdaMemElem<DDim>::run(ddc::DiscreteElement<DDim>(delems...))...));
    }

    template <class... ODDim>
    KOKKOS_FUNCTION constexpr auto operator[](
            ddc::DiscreteElement<ODDim...> const& slice_spec) const noexcept
    {
        // TODO static_assert unique mem_id
        return Tensor<
                ElementType,
                ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_remove_t<
                        ddc::detail::TypeSeq<DDim...>,
                        ddc::detail::TypeSeq<ODDim...>>>,
                LayoutStridedPolicy,
                MemorySpace>(ddc::ChunkSpan<
                             ElementType,
                             ddc::DiscreteDomain<DDim...>,
                             LayoutStridedPolicy,
                             MemorySpace>::
                             operator[](ddc::DiscreteElement<ODDim...>(
                                     detail::LambdaMemElem<ODDim>::run(slice_spec)...)));
    }

    Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>& operator+=(
            const Tensor<
                    ElementType,
                    ddc::DiscreteDomain<DDim...>,
                    LayoutStridedPolicy,
                    MemorySpace>& tensor)
    {
        ddc::for_each(this->domain(), [&](ddc::DiscreteElement<DDim...> elem) {
            (*this)(elem) += tensor(elem);
        });
        return *this;
    }

    Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>& operator*=(
            const ElementType scalar)
    {
        ddc::for_each(this->domain(), [&](ddc::DiscreteElement<DDim...> elem) {
            (*this)(elem) *= scalar;
        });
        return *this;
    }
};

// Relabelize index without altering allocation
namespace detail {
template <class IndexToRelabelize, class OldIndex, class NewIndex>
struct RelabelizeIndex
{
    using type = std::
            conditional_t<std::is_same_v<IndexToRelabelize, OldIndex>, NewIndex, IndexToRelabelize>;
};

template <
        template <class...>
        class IndexToRelabelizeType,
        class OldIndex,
        class NewIndex,
        class... Arg>
struct RelabelizeIndex<IndexToRelabelizeType<Arg...>, OldIndex, NewIndex>
{
    using type = std::conditional_t<
            TensorNatIndex<IndexToRelabelizeType<Arg...>>,
            std::conditional_t<
                    std::is_same_v<IndexToRelabelizeType<Arg...>, OldIndex>,
                    NewIndex,
                    IndexToRelabelizeType<Arg...>>,
            IndexToRelabelizeType<
                    std::conditional_t<std::is_same_v<Arg, OldIndex>, NewIndex, Arg>...>>;
};

template <class Dom, class OldIndex, class NewIndex>
struct RelabelizeIndexInDomainType;

template <class... DDim, class OldIndex, class NewIndex>
struct RelabelizeIndexInDomainType<ddc::DiscreteDomain<DDim...>, OldIndex, NewIndex>
{
    using type = ddc::DiscreteDomain<typename RelabelizeIndex<DDim, OldIndex, NewIndex>::type...>;
};

} // namespace detail

template <class Dom, class OldIndex, class NewIndex>
using relabelize_index_in_domain_t
        = detail::RelabelizeIndexInDomainType<Dom, OldIndex, NewIndex>::type;

namespace detail {
template <class TensorType, class OldIndex, class NewIndex>
struct RelabelizeIndexOfType;

template <
        class OldIndex,
        class NewIndex,
        class ElementType,
        class Dom,
        class LayoutStridedPolicy,
        class MemorySpace>
struct RelabelizeIndexOfType<
        Tensor<ElementType, Dom, LayoutStridedPolicy, MemorySpace>,
        OldIndex,
        NewIndex>
{
    using type = Tensor<
            ElementType,
            typename RelabelizeIndexInDomainType<Dom, OldIndex, NewIndex>::type,
            LayoutStridedPolicy,
            MemorySpace>;
};

} // namespace detail

// TODO relabelize_index_in_domain ?

template <class TensorType, class OldIndex, class NewIndex>
using relabelize_index_of_t = detail::RelabelizeIndexOfType<TensorType, OldIndex, NewIndex>::type;

template <
        class OldIndex,
        class NewIndex,
        class ElementType,
        class... DDim,
        class LayoutStridedPolicy,
        class MemorySpace>
relabelize_index_of_t<
        Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>,
        OldIndex,
        NewIndex>
relabelize_index_of(
        Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
                old_tensor)
{
    return relabelize_index_of_t<
            Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>,
            OldIndex,
            NewIndex>(
            old_tensor.data_handle(),
            typename detail::RelabelizeIndexInDomainType<
                    ddc::DiscreteDomain<DDim...>,
                    OldIndex,
                    NewIndex>::
                    type(ddc::DiscreteDomain<
                            typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::type>(
                            ddc::DiscreteElement<
                                    typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::
                                            type>(old_tensor.domain().front().template uid<DDim>()),
                            ddc::DiscreteVector<
                                    typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::
                                            type>(static_cast<std::size_t>(
                                    old_tensor.template extent<DDim>())))...));
}

namespace detail {
template <class IndexToRelabelize, class OldIndices, class NewIndices>
struct RelabelizeIndices;

template <template <class...> class IndexToRelabelizeType, class... Arg>
struct RelabelizeIndices<
        IndexToRelabelizeType<Arg...>,
        ddc::detail::TypeSeq<>,
        ddc::detail::TypeSeq<>>
{
    using type = IndexToRelabelizeType<Arg...>;
};

template <
        template <class...>
        class IndexToRelabelizeType,
        class HeadOldIndex,
        class... TailOldIndex,
        class HeadNewIndex,
        class... TailNewIndex,
        class... Arg>
struct RelabelizeIndices<
        IndexToRelabelizeType<Arg...>,
        ddc::detail::TypeSeq<HeadOldIndex, TailOldIndex...>,
        ddc::detail::TypeSeq<HeadNewIndex, TailNewIndex...>>
{
    using type = std::conditional_t<
            (sizeof...(TailOldIndex) > 0),
            typename RelabelizeIndices<
                    typename RelabelizeIndex<
                            IndexToRelabelizeType<Arg...>,
                            HeadOldIndex,
                            HeadNewIndex>::type,
                    ddc::detail::TypeSeq<TailOldIndex...>,
                    ddc::detail::TypeSeq<TailNewIndex...>>::type,
            typename RelabelizeIndex<IndexToRelabelizeType<Arg...>, HeadOldIndex, HeadNewIndex>::
                    type>;
};

template <class Dom, class OldIndices, class NewIndices>
struct RelabelizeIndicesInDomainType;

template <class Dom>
struct RelabelizeIndicesInDomainType<Dom, ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    using type = Dom;
};

template <
        class Dom,
        class HeadOldIndex,
        class... TailOldIndex,
        class HeadNewIndex,
        class... TailNewIndex>
struct RelabelizeIndicesInDomainType<
        Dom,
        ddc::detail::TypeSeq<HeadOldIndex, TailOldIndex...>,
        ddc::detail::TypeSeq<HeadNewIndex, TailNewIndex...>>
{
    using type = typename RelabelizeIndicesInDomainType<
            relabelize_index_in_domain_t<Dom, HeadOldIndex, HeadNewIndex>,
            ddc::detail::TypeSeq<TailOldIndex...>,
            ddc::detail::TypeSeq<TailNewIndex...>>::type;
};

} // namespace detail

template <class Dom, class OldIndices, class NewIndices>
using relabelize_indices_in_domain_t =
        typename detail::RelabelizeIndicesInDomainType<Dom, OldIndices, NewIndices>::type;

namespace detail {

template <class OldIndices, class NewIndices, std::size_t I>
struct RelabelizeIndicesInDomain
{
    template <class... DDim>
    static auto run(ddc::DiscreteDomain<DDim...> dom)
    {
        if constexpr (I != ddc::type_seq_size_v<OldIndices>) {
            return RelabelizeIndicesInDomain<OldIndices, NewIndices, I + 1>::run(
                    relabelize_index_in_domain_t<
                            ddc::DiscreteDomain<DDim...>,
                            ddc::type_seq_element_t<I, OldIndices>,
                            ddc::type_seq_element_t<I, NewIndices>>(
                            ddc::DiscreteDomain<typename detail::RelabelizeIndex<
                                    DDim,
                                    ddc::type_seq_element_t<I, OldIndices>,
                                    ddc::type_seq_element_t<I, NewIndices>>::type>(
                                    ddc::DiscreteElement<typename detail::RelabelizeIndex<
                                            DDim,
                                            ddc::type_seq_element_t<I, OldIndices>,
                                            ddc::type_seq_element_t<I, NewIndices>>::type>(
                                            dom.front().template uid<DDim>()),
                                    ddc::DiscreteVector<typename detail::RelabelizeIndex<
                                            DDim,
                                            ddc::type_seq_element_t<I, OldIndices>,
                                            ddc::type_seq_element_t<I, NewIndices>>::type>(
                                            static_cast<std::size_t>(
                                                    dom.template extent<DDim>())))...));
        } else {
            return dom;
        }
    }
};

} // namespace detail

template <class OldIndices, class NewIndices, class Dom>
relabelize_indices_in_domain_t<Dom, OldIndices, NewIndices> relabelize_indices_in_domain(Dom dom)
{
    return detail::RelabelizeIndicesInDomain<OldIndices, NewIndices, 0>::run(dom);
}

namespace detail {

template <class TensorType, class OldIndex, class NewIndex>
struct RelabelizeIndicesOfType;

template <
        class OldIndices,
        class NewIndices,
        class ElementType,
        class Dom,
        class LayoutStridedPolicy,
        class MemorySpace>
struct RelabelizeIndicesOfType<
        Tensor<ElementType, Dom, LayoutStridedPolicy, MemorySpace>,
        OldIndices,
        NewIndices>
{
    using type = Tensor<
            ElementType,
            typename RelabelizeIndicesInDomainType<Dom, OldIndices, NewIndices>::type,
            LayoutStridedPolicy,
            MemorySpace>;
};

template <
        class OldIndices,
        class NewIndices,
        std::size_t I,
        class ElementType,
        class... DDim,
        class LayoutStridedPolicy,
        class MemorySpace>
auto RelabelizeIndicesOf(
        Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
                old_tensor)
{
    if constexpr (I != ddc::type_seq_size_v<OldIndices>) {
        return RelabelizeIndicesOf<
                ddc::type_seq_replace_t<
                        OldIndices,
                        ddc::detail::TypeSeq<ddc::type_seq_element_t<I, OldIndices>>,
                        ddc::detail::TypeSeq<ddc::type_seq_element_t<I, NewIndices>>>,
                NewIndices,
                I + 1>(relabelize_index_of_t<
                       Tensor<ElementType,
                              ddc::DiscreteDomain<DDim...>,
                              LayoutStridedPolicy,
                              MemorySpace>,
                       ddc::type_seq_element_t<I, OldIndices>,
                       ddc::type_seq_element_t<I, NewIndices>>(
                old_tensor.data_handle(),
                typename detail::RelabelizeIndexInDomainType<
                        ddc::DiscreteDomain<DDim...>,
                        ddc::type_seq_element_t<I, OldIndices>,
                        ddc::type_seq_element_t<I, NewIndices>>::
                        type(ddc::DiscreteDomain<typename detail::RelabelizeIndex<
                                     DDim,
                                     ddc::type_seq_element_t<I, OldIndices>,
                                     ddc::type_seq_element_t<I, NewIndices>>::type>(
                                ddc::DiscreteElement<typename detail::RelabelizeIndex<
                                        DDim,
                                        ddc::type_seq_element_t<I, OldIndices>,
                                        ddc::type_seq_element_t<I, NewIndices>>::type>(
                                        old_tensor.domain().front().template uid<DDim>()),
                                ddc::DiscreteVector<typename detail::RelabelizeIndex<
                                        DDim,
                                        ddc::type_seq_element_t<I, OldIndices>,
                                        ddc::type_seq_element_t<I, NewIndices>>::type>(
                                        static_cast<std::size_t>(
                                                old_tensor.template extent<DDim>())))...)));
    } else {
        return old_tensor;
    }
}

} // namespace detail

template <class TensorType, class OldIndices, class NewIndices>
using relabelize_indices_of_t
        = detail::RelabelizeIndicesOfType<TensorType, OldIndices, NewIndices>::type;

template <class OldIndices, class NewIndices, class Tensor>
relabelize_indices_of_t<Tensor, OldIndices, NewIndices> relabelize_indices_of(Tensor tensor)
{
    return detail::RelabelizeIndicesOf<OldIndices, NewIndices, 0>(tensor);
}

// Sum of tensors
template <
        class... DDim,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace,
        class... TensorType>
Tensor<ElementType,
       ddc::DiscreteDomain<DDim...>,
       Kokkos::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_sum(
        Tensor<ElementType,
               ddc::DiscreteDomain<DDim...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> sum_tensor,
        TensorType... tensor)
{
    ddc::for_each(sum_tensor.domain(), [&](ddc::DiscreteElement<DDim...> elem) {
        sum_tensor(elem) = (tensor(elem) + ...);
    });
    return sum_tensor;
}

namespace detail {

// Domain of a tensor result of product between two tensors
template <class Dom1, class Dom2>
struct NaturalTensorProdDomain;

template <class... DDim1, class... DDim2>
struct NaturalTensorProdDomain<ddc::DiscreteDomain<DDim1...>, ddc::DiscreteDomain<DDim2...>>
{
    using type = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_merge_t<
            ddc::type_seq_remove_t<ddc::detail::TypeSeq<DDim1...>, ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<DDim2...>,
                    ddc::detail::TypeSeq<DDim1...>>>>;
};

} // namespace detail

template <class Dom1, class Dom2>
using natural_tensor_prod_domain_t = detail::NaturalTensorProdDomain<Dom1, Dom2>::type;

template <class Dom1, class Dom2>
natural_tensor_prod_domain_t<Dom1, Dom2> natural_tensor_prod_domain(Dom1 dom1, Dom2 dom2)
{
    return natural_tensor_prod_domain_t<Dom1, Dom2>(dom1, dom2);
}

namespace detail {

// Product between two tensors naturally indexed.
template <class HeadDDim1TypeSeq, class ContractDDimTypeSeq, class TailDDim2TypeSeq>
struct NaturalTensorProd;

template <class... HeadDDim1, class... ContractDDim, class... TailDDim2>
struct NaturalTensorProd<
        ddc::detail::TypeSeq<HeadDDim1...>,
        ddc::detail::TypeSeq<ContractDDim...>,
        ddc::detail::TypeSeq<TailDDim2...>>
{
    template <class ElementType, class LayoutStridedPolicy, class MemorySpace>
    static Tensor<
            ElementType,
            ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
            LayoutStridedPolicy,
            MemorySpace>
    run(Tensor<ElementType,
               ddc::DiscreteDomain<HeadDDim1..., TailDDim2...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType,
               ddc::DiscreteDomain<HeadDDim1..., ContractDDim...>,
               LayoutStridedPolicy,
               MemorySpace> tensor1,
        Tensor<ElementType,
               ddc::DiscreteDomain<ContractDDim..., TailDDim2...>,
               LayoutStridedPolicy,
               MemorySpace> tensor2)
    {
        ddc::for_each(
                prod_tensor.domain(),
                [&](ddc::DiscreteElement<HeadDDim1..., TailDDim2...> elem) {
                    prod_tensor(elem) = ddc::transform_reduce(
                            tensor1.template domain<ContractDDim...>(),
                            0.,
                            ddc::reducer::sum<ElementType>(),
                            [&](ddc::DiscreteElement<ContractDDim...> contract_elem) {
                                return tensor1(ddc::select<HeadDDim1...>(elem), contract_elem)
                                       * tensor2(ddc::select<TailDDim2...>(elem), contract_elem);
                            });
                });
        return prod_tensor;
    }
};

} // namespace detail

template <
        TensorNatIndex... ProdDDim,
        TensorNatIndex... DDim1,
        TensorNatIndex... DDim2,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace>
Tensor<ElementType,
       ddc::DiscreteDomain<ProdDDim...>,
       Kokkos::layout_right,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               Kokkos::layout_right,
               Kokkos::DefaultHostExecutionSpace::memory_space> prod_tensor,
        Tensor<ElementType, ddc::DiscreteDomain<DDim1...>, LayoutStridedPolicy, MemorySpace>
                tensor1,
        Tensor<ElementType, ddc::DiscreteDomain<DDim2...>, LayoutStridedPolicy, MemorySpace>
                tensor2)
{
    static_assert(std::is_same_v<
                  ddc::type_seq_remove_t<
                          ddc::detail::TypeSeq<DDim1...>,
                          ddc::detail::TypeSeq<ProdDDim...>>,
                  ddc::type_seq_remove_t<
                          ddc::detail::TypeSeq<DDim2...>,
                          ddc::detail::TypeSeq<ProdDDim...>>>);
    return detail::NaturalTensorProd<
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::detail::TypeSeq<DDim2...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<DDim1...>,
                    ddc::detail::TypeSeq<ProdDDim...>>,
            ddc::type_seq_remove_t<
                    ddc::detail::TypeSeq<ProdDDim...>,
                    ddc::detail::TypeSeq<DDim1...>>>::run(prod_tensor, tensor1, tensor2);
}

namespace detail {

template <class HeadDom, class InterestDom, class TailDom>
struct PrintTensor;

template <class... HeadDDim, class InterestDDim, class HeadOfTailDDim, class... TailOfTailDDim>
struct PrintTensor<
        ddc::DiscreteDomain<HeadDDim...>,
        ddc::DiscreteDomain<InterestDDim>,
        ddc::DiscreteDomain<HeadOfTailDDim, TailOfTailDDim...>>
{
    template <class TensorType>
    static std::string run(
            std::string& str,
            TensorType const& tensor,
            ddc::DiscreteElement<HeadDDim...> i)
    {
        str += "[";
        for (ddc::DiscreteElement<InterestDDim> elem :
             ddc::DiscreteDomain<InterestDDim>(tensor.natural_domain())) {
            str = PrintTensor<
                    ddc::DiscreteDomain<HeadDDim..., InterestDDim>,
                    ddc::DiscreteDomain<HeadOfTailDDim>,
                    ddc::DiscreteDomain<TailOfTailDDim...>>::
                    run(str, tensor, ddc::DiscreteElement<HeadDDim..., InterestDDim>(i, elem));
        }
        str += "]\n";
        return str;
    }
};

template <class... HeadDDim, class InterestDDim>
struct PrintTensor<
        ddc::DiscreteDomain<HeadDDim...>,
        ddc::DiscreteDomain<InterestDDim>,
        ddc::DiscreteDomain<>>
{
    template <class TensorType>
    static std::string run(
            std::string& str,
            TensorType const& tensor,
            ddc::DiscreteElement<HeadDDim...> i)
    {
        for (ddc::DiscreteElement<InterestDDim> elem :
             ddc::DiscreteDomain<InterestDDim>(tensor.natural_domain())) {
            str = str + " "
                  + std::to_string(tensor.get(tensor.access_element(
                          ddc::DiscreteElement<HeadDDim..., InterestDDim>(i, elem))));
        }
        str += "\n";
        return str;
    }
};

} // namespace detail

template <class ElementType, class... DDim, class LayoutStridedPolicy, class MemorySpace>
std::ostream& operator<<(
        std::ostream& os,
        Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace> const&
                tensor)
{
    std::string str = "";
    os << detail::PrintTensor<
            ddc::DiscreteDomain<>,
            ddc::DiscreteDomain<ddc::type_seq_element_t<
                    0,
                    ddc::to_type_seq_t<typename TensorAccessor<DDim...>::natural_domain_t>>>,
            ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename TensorAccessor<DDim...>::natural_domain_t>,
                    ddc::detail::TypeSeq<ddc::type_seq_element_t<
                            0,
                            ddc::to_type_seq_t<
                                    typename TensorAccessor<DDim...>::natural_domain_t>>>>>>::
                    run(str, tensor, ddc::DiscreteElement<>());
    return os;
}

} // namespace tensor

} // namespace sil
