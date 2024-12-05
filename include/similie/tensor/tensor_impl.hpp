// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <iostream>

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>

namespace sil {

namespace tensor {

// struct representing an index mu or nu in a tensor Tmunu.
template <class... CDim>
struct TensorNaturalIndex
{
    static constexpr bool is_tensor_index = true;
    static constexpr bool is_tensor_natural_index = true;
    static constexpr bool is_explicitely_stored_tensor = true;

    using type_seq_dimensions = ddc::detail::TypeSeq<CDim...>;

    using subindices_domain_t = ddc::DiscreteDomain<>;

    KOKKOS_FUNCTION static constexpr subindices_domain_t subindices_domain()
    {
        return ddc::DiscreteDomain<>();
    }

    KOKKOS_FUNCTION static constexpr std::size_t rank()
    {
        return sizeof...(CDim) != 0;
    }

    KOKKOS_FUNCTION static constexpr std::size_t size()
    {
        if constexpr (rank() == 0) {
            return 1;
        } else {
            return sizeof...(CDim);
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_size()
    {
        return size();
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_size()
    {
        return size();
    }

    template <class ODim>
    KOKKOS_FUNCTION static constexpr std::size_t mem_id()
    {
        if constexpr (rank() == 0) {
            return 0;
        } else {
            return ddc::type_seq_rank_v<ODim, type_seq_dimensions>;
        }
    }

    KOKKOS_FUNCTION static constexpr std::size_t mem_id(std::size_t const natural_id)
    {
        return natural_id;
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id(std::size_t const natural_id)
    {
        return natural_id;
    }

    KOKKOS_FUNCTION static constexpr std::size_t access_id_to_mem_id(std::size_t access_id)
    {
        return access_id;
    }

    template <class Tensor, class Elem, class Id>
    KOKKOS_FUNCTION static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        return access(tensor, elem);
    }

    KOKKOS_FUNCTION static constexpr std::array<std::size_t, rank()>
    mem_id_to_canonical_natural_ids(std::size_t mem_id)
    {
        assert(mem_id < mem_size());
        if constexpr (rank() == 0) {
            return std::array<std::size_t, rank()> {};
        } else {
            return std::array<std::size_t, rank()> {mem_id};
        }
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

// natural_domain_t is obtained using concept and specialization
namespace detail {

template <class Index>
struct NaturalDomainType;

template <class Index>
    requires(TensorIndex<Index> && !TensorNatIndex<Index>)
struct NaturalDomainType<Index>
{
    using type = typename Index::subindices_domain_t;
};

template <TensorNatIndex Index>
struct NaturalDomainType<Index>
{
    using type = ddc::DiscreteDomain<Index>;
};

} // namespace detail

template <TensorIndex Index>
using natural_domain_t = typename detail::NaturalDomainType<Index>::type;

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
static constexpr std::size_t access_id()
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
static constexpr std::size_t access_id(ddc::DiscreteElement<NaturalIndex...> natural_elem)
{
    if constexpr (TensorNatIndex<Index>) {
        return IdFromElem<Index, ddc::DiscreteDomain<Index>>::run(natural_elem);
    } else {
        return IdFromElem<Index, typename Index::subindices_domain_t>::run(natural_elem);
    }
}

} // namespace detail

// TensorAccessor class, allows to build a domain which represents the tensor and access elements.
template <TensorIndex... Index>
class TensorAccessor
{
public:
    explicit constexpr TensorAccessor();

    using discrete_domain_type = ddc::DiscreteDomain<Index...>;

    using discrete_element_type = ddc::DiscreteElement<Index...>;

    using natural_domain_t = ddc::cartesian_prod_t<std::conditional_t< // TODO natural_domain_type
            TensorNatIndex<Index>,
            ddc::DiscreteDomain<Index>,
            typename Index::subindices_domain_t>...>;

    static constexpr natural_domain_t natural_domain();

    static constexpr discrete_domain_type mem_domain();

    static constexpr discrete_domain_type access_domain();

    template <class... CDim>
    static constexpr discrete_element_type access_element();

    template <class... NaturalIndex>
    static constexpr discrete_element_type access_element(
            ddc::DiscreteElement<NaturalIndex...> natural_elem);

    template <class... MemIndex>
    static constexpr natural_domain_t::discrete_element_type canonical_natural_element(
            ddc::DiscreteElement<MemIndex...> mem_elem);
};

namespace detail {

template <class Seq>
struct TensorAccessorForTypeSeq;

template <TensorIndex... Index>
struct TensorAccessorForTypeSeq<ddc::detail::TypeSeq<Index...>>
{
    using type = TensorAccessor<Index...>;
};

template <class Dom>
struct TensorAccessorForDomain;

template <class... DDim>
struct TensorAccessorForDomain<ddc::DiscreteDomain<DDim...>>
{
    using type = typename TensorAccessorForTypeSeq<
            ddc::to_type_seq_t<ddc::cartesian_prod_t<std::conditional_t<
                    TensorIndex<DDim>,
                    ddc::DiscreteDomain<DDim>,
                    ddc::DiscreteDomain<>>...>>>::type;
};

} // namespace detail

template <misc::Specialization<ddc::DiscreteDomain> Dom>
using tensor_accessor_for_domain_t = detail::TensorAccessorForDomain<Dom>::type;

template <TensorIndex... Index>
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

template <TensorIndex... Index>
constexpr TensorAccessor<Index...>::natural_domain_t TensorAccessor<Index...>::natural_domain()
{
    return natural_domain_t(detail::natural_domain<Index>()...);
}

template <TensorIndex... Index>
constexpr TensorAccessor<Index...>::discrete_domain_type TensorAccessor<Index...>::mem_domain()
{
    return ddc::DiscreteDomain<Index...>(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::mem_size())...));
}

template <TensorIndex... Index>
constexpr TensorAccessor<Index...>::discrete_domain_type TensorAccessor<Index...>::access_domain()
{
    return ddc::DiscreteDomain<Index...>(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::access_size())...));
}

template <TensorIndex... Index>
template <class... CDim>
constexpr TensorAccessor<Index...>::discrete_element_type TensorAccessor<Index...>::access_element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            detail::access_id<Index, ddc::detail::TypeSeq<Index...>, CDim...>())...);
}

template <TensorIndex... Index>
template <class... NaturalIndex>
constexpr TensorAccessor<Index...>::discrete_element_type TensorAccessor<Index...>::access_element(
        ddc::DiscreteElement<NaturalIndex...> natural_elem)
{
    return ddc::DiscreteElement<Index...>(
            ddc::DiscreteElement<Index>(detail::access_id<Index, ddc::detail::TypeSeq<Index...>>(
                    typename natural_domain_t::discrete_element_type(natural_elem)))...);
}

template <TensorIndex... Index>
template <class... MemIndex>
constexpr TensorAccessor<Index...>::natural_domain_t::discrete_element_type TensorAccessor<
        Index...>::canonical_natural_element(ddc::DiscreteElement<MemIndex...> mem_elem)
{
    std::array<std::size_t, natural_domain_t::rank()> ids {};
    auto it = ids.begin();
    (
            [&]() {
                auto i = MemIndex::mem_id_to_canonical_natural_ids(
                        mem_elem.template uid<MemIndex>());
                std::copy(i.begin(), i.end(), it);
                it += i.size();
            }(),
            ...);
    typename natural_domain_t::discrete_element_type natural_elem;
    ddc::detail::array(natural_elem) = std::array<std::size_t, natural_domain_t::rank()>(ids);
    return natural_elem;
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
    KOKKOS_FUNCTION static TensorField::element_type run(TensorField tensor_field, Elem const& elem)
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
                        KOKKOS_LAMBDA(TensorField tensor_field_, Elem elem_)
                                ->TensorField::element_type {
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
                        KOKKOS_LAMBDA(TensorField tensor_field_, Elem elem_)
                                ->TensorField::element_type {
                                    double tensor_field_value = 0;
                                    if constexpr (IndexInterest::is_explicitely_stored_tensor) {
                                        std::size_t const mem_id
                                                = IndexInterest::access_id_to_mem_id(
                                                        elem_.template uid<IndexInterest>());
                                        if (mem_id != std::numeric_limits<std::size_t>::max()) {
                                            tensor_field_value
                                                    = tensor_field_
                                                              .mem(ddc::DiscreteElement<
                                                                           IndexHead...>(elem_),
                                                                   ddc::DiscreteElement<
                                                                           IndexInterest>(mem_id));
                                        } else {
                                            tensor_field_value = 1.;
                                        }
                                    } else {
                                        std::pair<
                                                std::vector<double>,
                                                std::vector<std::size_t>> const mem_lin_comb
                                                = IndexInterest::access_id_to_mem_lin_comb(
                                                        elem_.template uid<IndexInterest>());

                                        if (std::get<0>(mem_lin_comb).size() > 0) {
                                            for (std::size_t i = 0;
                                                 i < std::get<0>(mem_lin_comb).size();
                                                 ++i) {
                                                tensor_field_value
                                                        += std::get<0>(mem_lin_comb)[i]
                                                           * tensor_field_.mem(
                                                                   ddc::DiscreteElement<
                                                                           IndexHead...>(elem_),
                                                                   ddc::DiscreteElement<
                                                                           IndexInterest>(std::get<
                                                                                          1>(
                                                                           mem_lin_comb)[i]));
                                            }
                                        } else {
                                            tensor_field_value = 1.;
                                        }
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
template <class InterestDim>
struct LambdaMemElem
{
    template <class Elem>
    KOKKOS_FUNCTION static ddc::DiscreteElement<InterestDim> run(Elem elem)
    {
        return ddc::DiscreteElement<InterestDim>(elem);
    }
};

template <TensorIndex InterestDim>
struct LambdaMemElem<InterestDim>
{
    template <class Elem>
    KOKKOS_FUNCTION static ddc::DiscreteElement<InterestDim> run(Elem elem)
    {
        if constexpr (InterestDim::is_explicitely_stored_tensor) {
            std::size_t const mem_id
                    = InterestDim::access_id_to_mem_id(elem.template uid<InterestDim>());
            assert(mem_id != std::numeric_limits<std::size_t>::max()
                   && "mem_elem is not defined because mem_id() returned a max integer");
            return ddc::DiscreteElement<InterestDim>(mem_id);
        } else {
            std::pair<std::vector<double>, std::vector<std::size_t>> const mem_lin_comb
                    = InterestDim::access_id_to_mem_lin_comb(elem.template uid<InterestDim>());
            assert(std::get<0>(mem_lin_comb).size() > 0
                   && "mem_elem is not defined because mem_lin_comb contains no id");
            assert(std::get<0>(mem_lin_comb).size() == 1
                   && "mem_elem is not defined because mem_lin_comb contains several ids");
            return ddc::DiscreteElement<InterestDim>(std::get<1>(mem_lin_comb)[0]);
        }
    }
};

} // namespace detail

// @cond

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class Tensor;

} // namespace tensor

} // namespace sil

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

namespace sil {

namespace tensor {

// @endcond

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
    using base_type::ChunkSpan;
    using reference = base_type::reference;
    using discrete_domain_type = base_type::discrete_domain_type;
    using discrete_element_type = base_type::discrete_element_type;

    using base_type::domain;
    using base_type::operator();

    KOKKOS_FUNCTION constexpr explicit Tensor(ddc::ChunkSpan<
                                              ElementType,
                                              ddc::DiscreteDomain<DDim...>,
                                              LayoutStridedPolicy,
                                              MemorySpace> other) noexcept
        : base_type(other)
    {
    }

    using accessor_t = tensor_accessor_for_domain_t<ddc::cartesian_prod_t<std::conditional_t<
            TensorIndex<DDim>,
            ddc::DiscreteDomain<DDim>,
            ddc::DiscreteDomain<>>...>>;

    static constexpr accessor_t accessor()
    {
        return accessor_t();
    }

    using indices_domain_t = accessor_t::discrete_domain_type;

    using non_indices_domain_t
            = ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<discrete_domain_type>,
                    ddc::to_type_seq_t<indices_domain_t>>>;

    KOKKOS_FUNCTION constexpr indices_domain_t indices_domain() const noexcept
    {
        return indices_domain_t(domain());
    }

    KOKKOS_FUNCTION constexpr non_indices_domain_t non_indices_domain() const noexcept
    {
        return non_indices_domain_t(domain());
    }

    using natural_domain_t
            = ddc::cartesian_prod_t<non_indices_domain_t, typename accessor_t::natural_domain_t>;

    KOKKOS_FUNCTION constexpr natural_domain_t natural_domain() const noexcept
    {
        return natural_domain_t(non_indices_domain(), accessor_t::natural_domain());
    }

    KOKKOS_FUNCTION constexpr discrete_domain_type mem_domain() const noexcept
    {
        return discrete_domain_type(non_indices_domain(), accessor_t::mem_domain());
    }

    KOKKOS_FUNCTION constexpr discrete_domain_type access_domain() const noexcept
    {
        return discrete_domain_type(non_indices_domain(), accessor_t::access_domain());
    }

    template <class... CDim>
    KOKKOS_FUNCTION constexpr discrete_element_type access_element()
            const noexcept // TODO merge this with the one below
    {
        return discrete_element_type(accessor_t::template access_element<CDim...>());
    }

    template <class... Elem>
    KOKKOS_FUNCTION constexpr discrete_element_type access_element(Elem... elem) const noexcept
    {
        return discrete_element_type(
                accessor_t::access_element(
                        typename accessor_t::natural_domain_t::discrete_element_type(elem...)),
                typename non_indices_domain_t::discrete_element_type(elem...));
    }

    template <class... Elem>
    KOKKOS_FUNCTION constexpr natural_domain_t::discrete_element_type canonical_natural_element(
            Elem... mem_elem) const noexcept
    {
        return typename natural_domain_t::discrete_element_type(
                accessor_t::canonical_natural_element(
                        typename accessor_t::discrete_element_type(mem_elem...)),
                typename non_indices_domain_t::discrete_element_type(mem_elem...));
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
        ddc::ChunkSpan chunkspan = ddc::ChunkSpan<
                ElementType,
                ddc::DiscreteDomain<DDim...>,
                LayoutStridedPolicy,
                MemorySpace>::
        operator[](
                ddc::DiscreteElement<ODDim...>(detail::LambdaMemElem<ODDim>::run(slice_spec)...));
        return Tensor<
                ElementType,
                ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_remove_t<
                        ddc::detail::TypeSeq<DDim...>,
                        ddc::detail::TypeSeq<ODDim...>>>,
                typename decltype(chunkspan)::layout_type,
                MemorySpace>(chunkspan);
    }

    template <class... DElems>
    KOKKOS_FUNCTION ElementType get(DElems const&... delems) const noexcept
    {
        if constexpr (sizeof...(DDim) == 0) {
            return operator()(delems...);
        } else {
            return detail::Access<
                    Tensor<ElementType,
                           ddc::DiscreteDomain<DDim...>,
                           LayoutStridedPolicy,
                           MemorySpace>,
                    ddc::DiscreteElement<DDim...>,
                    ddc::detail::TypeSeq<>,
                    DDim...>::run(*this, ddc::DiscreteElement<DDim...>(delems...));
        }
    }

    Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>& operator+=(
            const Tensor<
                    ElementType,
                    ddc::DiscreteDomain<DDim...>,
                    LayoutStridedPolicy,
                    MemorySpace>& tensor)
    {
        ddc::for_each(this->domain(), [&](ddc::DiscreteElement<DDim...> elem) {
            this->mem(elem) += tensor.mem(elem);
        });
        return *this;
    }

    Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>& operator*=(
            const ElementType scalar)
    {
        ddc::for_each(this->domain(), [&](ddc::DiscreteElement<DDim...> elem) {
            this->mem(elem) *= scalar;
        });
        return *this;
    }
};

// Relabelize index without altering allocation
namespace detail {
template <class IndexToRelabelize, TensorIndex OldIndex, TensorIndex NewIndex>
struct RelabelizeIndex;

template <class IndexToRelabelize, TensorIndex OldIndex, TensorIndex NewIndex>
    requires(!TensorIndex<IndexToRelabelize> || TensorNatIndex<IndexToRelabelize>)
struct RelabelizeIndex<IndexToRelabelize, OldIndex, NewIndex>
{
    using type = std::
            conditional_t<std::is_same_v<IndexToRelabelize, OldIndex>, NewIndex, IndexToRelabelize>;
};

template <
        template <class...>
        class IndexToRelabelizeType,
        TensorIndex OldIndex,
        TensorIndex NewIndex,
        class... Arg>
struct RelabelizeIndex<IndexToRelabelizeType<Arg...>, OldIndex, NewIndex>
{
    using type = std::conditional_t<
            std::is_same_v<IndexToRelabelizeType<Arg...>, OldIndex>,
            NewIndex,
            IndexToRelabelizeType<typename RelabelizeIndex<Arg, OldIndex, NewIndex>::type...>>;
};

template <class T, class OldIndex, class NewIndex>
struct RelabelizeIndexInType;

template <template <class...> class T, class... DDim, class OldIndex, class NewIndex>
struct RelabelizeIndexInType<T<DDim...>, OldIndex, NewIndex>
{
    using type = T<typename RelabelizeIndex<DDim, OldIndex, NewIndex>::type...>;
};

} // namespace detail

template <class T, TensorIndex OldIndex, TensorIndex NewIndex>
using relabelize_index_in_t = detail::RelabelizeIndexInType<T, OldIndex, NewIndex>::type;

namespace detail {

template <class OldIndex, class NewIndex>
struct RelabelizeIndexIn
{
    template <class... DDim>
    static auto run(ddc::DiscreteElement<DDim...> elem)
    {
        return ddc::DiscreteElement<
                typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::type...>(
                elem.template uid<DDim>()...);
    }

    template <class... DDim>
    static auto run(ddc::DiscreteVector<DDim...> vect)
    {
        return ddc::DiscreteVector<
                typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::type...>(
                static_cast<std::size_t>(vect.template get<DDim>())...);
    }

    template <class... DDim>
    static auto run(ddc::DiscreteDomain<DDim...> dom)
    {
        return relabelize_index_in_t<ddc::DiscreteDomain<DDim...>, OldIndex, NewIndex>(
                relabelize_index_in<OldIndex, NewIndex>(dom.front()),
                relabelize_index_in<OldIndex, NewIndex>(dom.extents()));
    }
};

} // namespace detail

template <class OldIndex, class NewIndex, class T>
relabelize_index_in_t<T, OldIndex, NewIndex> relabelize_index_in(T t)
{
    return detail::RelabelizeIndexIn<OldIndex, NewIndex>::run(t);
}

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
            typename RelabelizeIndexInType<Dom, OldIndex, NewIndex>::type,
            LayoutStridedPolicy,
            MemorySpace>;
};

} // namespace detail

template <misc::Specialization<Tensor> TensorType, TensorIndex OldIndex, TensorIndex NewIndex>
using relabelize_index_of_t = detail::RelabelizeIndexOfType<TensorType, OldIndex, NewIndex>::type;

template <
        TensorIndex OldIndex,
        TensorIndex NewIndex,
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
            typename detail::RelabelizeIndexInType<
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

template <class IndexToRelabelize>
struct RelabelizeIndices<IndexToRelabelize, ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    using type = IndexToRelabelize;
};

template <
        class IndexToRelabelize,
        class HeadOldIndex,
        class... TailOldIndex,
        class HeadNewIndex,
        class... TailNewIndex>
struct RelabelizeIndices<
        IndexToRelabelize,
        ddc::detail::TypeSeq<HeadOldIndex, TailOldIndex...>,
        ddc::detail::TypeSeq<HeadNewIndex, TailNewIndex...>>
{
    static_assert(sizeof...(TailOldIndex) == sizeof...(TailNewIndex));
    using type = std::conditional_t<
            (sizeof...(TailOldIndex) > 0),
            typename RelabelizeIndices<
                    typename RelabelizeIndex<IndexToRelabelize, HeadOldIndex, HeadNewIndex>::type,
                    ddc::detail::TypeSeq<TailOldIndex...>,
                    ddc::detail::TypeSeq<TailNewIndex...>>::type,
            typename RelabelizeIndex<IndexToRelabelize, HeadOldIndex, HeadNewIndex>::type>;
};

template <class T, class OldIndices, class NewIndices>
struct RelabelizeIndicesInType;

template <class T>
struct RelabelizeIndicesInType<T, ddc::detail::TypeSeq<>, ddc::detail::TypeSeq<>>
{
    using type = T;
};

template <
        class T,
        class HeadOldIndex,
        class... TailOldIndex,
        class HeadNewIndex,
        class... TailNewIndex>
struct RelabelizeIndicesInType<
        T,
        ddc::detail::TypeSeq<HeadOldIndex, TailOldIndex...>,
        ddc::detail::TypeSeq<HeadNewIndex, TailNewIndex...>>
{
    static_assert(sizeof...(TailOldIndex) == sizeof...(TailNewIndex));
    using type = typename RelabelizeIndicesInType<
            relabelize_index_in_t<T, HeadOldIndex, HeadNewIndex>,
            ddc::detail::TypeSeq<TailOldIndex...>,
            ddc::detail::TypeSeq<TailNewIndex...>>::type;
};

} // namespace detail

template <class T, class OldIndices, class NewIndices>
using relabelize_indices_in_t =
        typename detail::RelabelizeIndicesInType<T, OldIndices, NewIndices>::type;

namespace detail {

template <class OldIndices, class NewIndices, std::size_t I = 0>
struct RelabelizeIndicesIn
{
    template <class... DDim>
    static auto run(ddc::DiscreteElement<DDim...> elem)
    {
        if constexpr (I != ddc::type_seq_size_v<OldIndices>) {
            return RelabelizeIndicesIn<OldIndices, NewIndices, I + 1>::run(
                    relabelize_index_in<
                            ddc::type_seq_element_t<I, OldIndices>,
                            ddc::type_seq_element_t<I, NewIndices>>(elem));
        } else {
            return elem;
        }
    }

    template <class... DDim>
    static auto run(ddc::DiscreteVector<DDim...> vect)
    {
        if constexpr (I != ddc::type_seq_size_v<OldIndices>) {
            return RelabelizeIndicesIn<OldIndices, NewIndices, I + 1>::run(
                    relabelize_index_in<
                            ddc::type_seq_element_t<I, OldIndices>,
                            ddc::type_seq_element_t<I, NewIndices>>(vect));
        } else {
            return vect;
        }
    }

    template <class... DDim>
    static auto run(ddc::DiscreteDomain<DDim...> dom)
    {
        if constexpr (I != ddc::type_seq_size_v<OldIndices>) {
            return RelabelizeIndicesIn<OldIndices, NewIndices, I + 1>::run(
                    relabelize_index_in<
                            ddc::type_seq_element_t<I, OldIndices>,
                            ddc::type_seq_element_t<I, NewIndices>>(dom));
        } else {
            return dom;
        }
    }
};

} // namespace detail

template <class OldIndices, class NewIndices, class T>
relabelize_indices_in_t<T, OldIndices, NewIndices> relabelize_indices_in(T t)
{
    static_assert(ddc::type_seq_size_v<OldIndices> == ddc::type_seq_size_v<NewIndices>);
    return detail::RelabelizeIndicesIn<OldIndices, NewIndices>::run(t);
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
    static_assert(ddc::type_seq_size_v<OldIndices> == ddc::type_seq_size_v<NewIndices>);
    using type = Tensor<
            ElementType,
            typename RelabelizeIndicesInType<Dom, OldIndices, NewIndices>::type,
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
    static_assert(ddc::type_seq_size_v<OldIndices> == ddc::type_seq_size_v<NewIndices>);
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
                typename detail::RelabelizeIndexInType<
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

template <
        misc::Specialization<Tensor> TensorType,
        misc::Specialization<ddc::detail::TypeSeq> OldIndices,
        misc::Specialization<ddc::detail::TypeSeq> NewIndices>
using relabelize_indices_of_t
        = detail::RelabelizeIndicesOfType<TensorType, OldIndices, NewIndices>::type;

template <
        misc::Specialization<ddc::detail::TypeSeq> OldIndices,
        misc::Specialization<ddc::detail::TypeSeq> NewIndices,
        misc::Specialization<Tensor> Tensor>
relabelize_indices_of_t<Tensor, OldIndices, NewIndices> relabelize_indices_of(Tensor tensor)
{
    static_assert(ddc::type_seq_size_v<OldIndices> == ddc::type_seq_size_v<NewIndices>);
    return detail::RelabelizeIndicesOf<OldIndices, NewIndices, 0>(tensor);
}

// Sum of tensors
template <
        class... DDim,
        class ElementType,
        class LayoutStridedPolicy,
        class MemorySpace,
        misc::Specialization<Tensor>... TensorType>
Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace> tensor_sum(
        Tensor<ElementType, ddc::DiscreteDomain<DDim...>, LayoutStridedPolicy, MemorySpace>
                sum_tensor,
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

template <
        misc::Specialization<ddc::DiscreteDomain> Dom1,
        misc::Specialization<ddc::DiscreteDomain> Dom2>
using natural_tensor_prod_domain_t = detail::NaturalTensorProdDomain<Dom1, Dom2>::type;

template <
        misc::Specialization<ddc::DiscreteDomain> Dom1,
        misc::Specialization<ddc::DiscreteDomain> Dom2>
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
               LayoutStridedPolicy,
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
       LayoutStridedPolicy,
       Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        Tensor<ElementType,
               ddc::DiscreteDomain<ProdDDim...>,
               LayoutStridedPolicy,
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

} // namespace detail

template <misc::Specialization<Tensor> TensorType>
std::ostream& operator<<(std::ostream& os, TensorType const& tensor)
{
    std::string str = "";
    os << detail::PrintTensor<
            ddc::DiscreteDomain<>,
            ddc::DiscreteDomain<ddc::type_seq_element_t<
                    0,
                    ddc::to_type_seq_t<typename TensorType::natural_domain_t>>>,
            ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_remove_t<
                    ddc::to_type_seq_t<typename TensorType::natural_domain_t>,
                    ddc::detail::TypeSeq<ddc::type_seq_element_t<
                            0,
                            ddc::to_type_seq_t<typename TensorType::natural_domain_t>>>>>>::
                    run(str, tensor, ddc::DiscreteElement<>());
    return os;
}

} // namespace tensor

} // namespace sil
