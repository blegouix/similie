// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>
#include <similie/tensor/tensor_impl.hpp>

namespace sil {

namespace tensor {

struct Covariant
{
};

struct Contravariant
{
};

namespace detail {

template <class NaturalIndices>
struct TensorNaturalIndexFromTypeSeqDim;

template <class... CDim>
struct TensorNaturalIndexFromTypeSeqDim<ddc::detail::TypeSeq<CDim...>>
{
    using type = TensorNaturalIndex<CDim...>;
};

} // namespace detail

// struct representing an index mu or nu in a tensor Tmunu.
template <TensorNatIndex NaturalIndex>
struct TensorCovariantNaturalIndex
    : detail::TensorNaturalIndexFromTypeSeqDim<typename NaturalIndex::type_seq_dimensions>::type
{
    using character = Covariant;
};

template <TensorNatIndex NaturalIndex>
struct TensorContravariantNaturalIndex
    : detail::TensorNaturalIndexFromTypeSeqDim<typename NaturalIndex::type_seq_dimensions>::type
{
    using character = Contravariant;
};

// helpes to lower, upper or uncharacterize indices
namespace detail {

template <class Index>
struct Lower;

template <TensorNatIndex NaturalIndex>
struct Lower<TensorCovariantNaturalIndex<NaturalIndex>>
{
    using type = TensorCovariantNaturalIndex<NaturalIndex>;
};

template <TensorNatIndex NaturalIndex>
struct Lower<TensorContravariantNaturalIndex<NaturalIndex>>
{
    using type = TensorCovariantNaturalIndex<NaturalIndex>;
};

template <template <TensorIndex...> class T, TensorIndex... Index>
struct Lower<T<Index...>>
{
    using type = T<typename Lower<Index>::type...>;
};

} // namespace detail

template <class T>
using lower = detail::Lower<T>::type;

namespace detail {

template <class Index>
struct Upper;

template <TensorNatIndex NaturalIndex>
struct Upper<TensorCovariantNaturalIndex<NaturalIndex>>
{
    using type = TensorContravariantNaturalIndex<NaturalIndex>;
};

template <TensorNatIndex NaturalIndex>
struct Upper<TensorContravariantNaturalIndex<NaturalIndex>>
{
    using type = TensorContravariantNaturalIndex<NaturalIndex>;
};

template <template <TensorIndex...> class T, TensorIndex... Index>
struct Upper<T<Index...>>
{
    using type = T<typename Upper<Index>::type...>;
};

} // namespace detail

template <class T>
using upper = detail::Upper<T>::type;

namespace detail {

template <class Index>
struct SwapCharacter;

template <TensorNatIndex NaturalIndex>
struct SwapCharacter<TensorCovariantNaturalIndex<NaturalIndex>>
{
    using type = TensorContravariantNaturalIndex<NaturalIndex>;
};

template <TensorNatIndex NaturalIndex>
struct SwapCharacter<TensorContravariantNaturalIndex<NaturalIndex>>
{
    using type = TensorCovariantNaturalIndex<NaturalIndex>;
};

template <template <TensorIndex...> class T, TensorIndex... Index>
struct SwapCharacter<T<Index...>>
{
    using type = T<typename SwapCharacter<Index>::type...>;
};

} // namespace detail

template <class T>
using swap_character = detail::SwapCharacter<T>::type;

namespace detail {

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

template <template <TensorIndex...> class T, TensorIndex... Index>
struct Uncharacterize<T<Index...>>
{
    using type = T<typename Uncharacterize<Index>::type...>;
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
template <misc::Specialization<Tensor> TensorType>
using uncharacterize_tensor_t = relabelize_indices_of_t<
        TensorType,
        ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>,
        uncharacterize<ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>>>;

template <misc::Specialization<Tensor> TensorType>
uncharacterize_tensor_t<TensorType> uncharacterize_tensor(TensorType tensor)
{
    return relabelize_indices_of<
            ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>,
            uncharacterize<ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>>>(
            tensor);
}

} // namespace tensor

} // namespace sil
