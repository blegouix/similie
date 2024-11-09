// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

struct Covariant
{
};

struct Contravariant
{
};

namespace detail {

template <class NaturalIndex>
struct TensorNaturalIndexFromTypeSeqDim;

template <class... CDim>
struct TensorNaturalIndexFromTypeSeqDim<ddc::detail::TypeSeq<CDim...>>
{
    using type = TensorNaturalIndex<CDim...>;
};

} // namespace detail

// struct representing an index mu or nu in a tensor Tmunu.
template <class NaturalIndex>
struct TensorCovariantNaturalIndex
    : detail::TensorNaturalIndexFromTypeSeqDim<typename NaturalIndex::type_seq_dimensions>::type
{
    using character = Covariant;
};

template <class NaturalIndex>
struct TensorContravariantNaturalIndex
    : detail::TensorNaturalIndexFromTypeSeqDim<typename NaturalIndex::type_seq_dimensions>::type
{
    using character = Contravariant;
};

// helpes to lower, upper or uncharacterize indices
namespace detail {

template <class Index>
struct Lower;

template <class NaturalIndex>
struct Lower<TensorContravariantNaturalIndex<NaturalIndex>>
{
    using type = TensorCovariantNaturalIndex<NaturalIndex>;
};

} // namespace detail

template <class Index>
using lower = detail::Lower<Index>::type;

namespace detail {

template <class Index>
struct Upper;

template <class NaturalIndex>
struct Upper<TensorCovariantNaturalIndex<NaturalIndex>>
{
    using type = TensorContravariantNaturalIndex<NaturalIndex>;
};

} // namespace detail

namespace detail {

template <class Index>
using upper = detail::Upper<Index>::type;

template <class Index>
struct Uncharacterize;

template <class NaturalIndex>
struct Uncharacterize<TensorCovariantNaturalIndex<NaturalIndex>>
{
    using type = NaturalIndex;
};

template <class NaturalIndex>
struct Uncharacterize<TensorContravariantNaturalIndex<NaturalIndex>>
{
    using type = NaturalIndex;
};

template <class... Index>
struct Uncharacterize<ddc::detail::TypeSeq<Index...>>
{
    using type = ddc::detail::TypeSeq<typename Uncharacterize<Index>::type...>;
};

template <class Index>
struct Uncharacterize
{
    using type = detail::RelabelizeIndices<
            Index,
            ddc::to_type_seq_t<typename Index::subindices_domain_t>,
            typename Uncharacterize<
                    ddc::to_type_seq_t<typename Index::subindices_domain_t>>::type>::type;
};

} // namespace detail

template <class Index>
using uncharacterize = detail::Uncharacterize<Index>::type;

// uncharacterize a tensor
template <class TensorType>
using uncharacterize_tensor_t = relabelize_indices_of_t<
        TensorType,
        ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>,
        uncharacterize<ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>>>;

template <class TensorType>
uncharacterize_tensor_t<TensorType> uncharacterize_tensor(TensorType tensor)
{
    return relabelize_indices_of<
            ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>,
            uncharacterize<ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>>>(
            tensor);
}

} // namespace tensor

} // namespace sil
