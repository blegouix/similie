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
constexpr uncharacterize_tensor_t<TensorType> uncharacterize_tensor(TensorType tensor)
{
    return relabelize_indices_of<
            ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>,
            uncharacterize<ddc::to_type_seq_t<typename TensorType::accessor_t::natural_domain_t>>>(
            tensor);
}

// check if index is covariant
namespace detail {

template <TensorNatIndex Index>
struct IsCovariant;

template <TensorNatIndex Index>
struct IsCovariant<TensorCovariantNaturalIndex<Index>>
{
    static constexpr bool value = true;
};

template <TensorNatIndex Index>
struct IsCovariant<TensorContravariantNaturalIndex<Index>>
{
    static constexpr bool value = false;
};

} // namespace detail

template <class T>
bool constexpr is_covariant_v = detail::IsCovariant<T>::value;

namespace detail {

template <class Seq>
struct AreCovariant;

template <TensorNatIndex... Index>
struct AreCovariant<ddc::detail::TypeSeq<Index...>>
{
    static constexpr bool value = (is_covariant_v<Index> && ...);
};

} // namespace detail

template <misc::Specialization<ddc::detail::TypeSeq> Seq>
bool constexpr are_covariant_v = detail::AreCovariant<Seq>::value;

// check if index is contravariant
namespace detail {

template <TensorNatIndex Index>
struct IsContravariant;

template <TensorNatIndex Index>
struct IsContravariant<TensorCovariantNaturalIndex<Index>>
{
    static constexpr bool value = false;
};

template <TensorNatIndex Index>
struct IsContravariant<TensorContravariantNaturalIndex<Index>>
{
    static constexpr bool value = true;
};

} // namespace detail

template <class T>
bool constexpr is_contravariant_v = detail::IsContravariant<T>::value;

namespace detail {

template <class Seq>
struct AreContravariant;

template <TensorNatIndex... Index>
struct AreContravariant<ddc::detail::TypeSeq<Index...>>
{
    static constexpr bool value = (is_contravariant_v<Index> && ...);
};

} // namespace detail

template <misc::Specialization<ddc::detail::TypeSeq> Seq>
bool constexpr are_contravariant_v = detail::AreContravariant<Seq>::value;

// check if characters are equal
namespace detail {

template <TensorNatIndex Index1, TensorNatIndex Index2>
struct IsSameCharacter;

template <TensorNatIndex Index1, TensorNatIndex Index2>
struct IsSameCharacter<TensorCovariantNaturalIndex<Index1>, TensorCovariantNaturalIndex<Index2>>
{
    static constexpr bool value = true;
};

template <TensorNatIndex Index1, TensorNatIndex Index2>
struct IsSameCharacter<TensorContravariantNaturalIndex<Index1>, TensorCovariantNaturalIndex<Index2>>
{
    static constexpr bool value = false;
};

template <TensorNatIndex Index1, TensorNatIndex Index2>
struct IsSameCharacter<TensorCovariantNaturalIndex<Index1>, TensorContravariantNaturalIndex<Index2>>
{
    static constexpr bool value = false;
};

template <TensorNatIndex Index1, TensorNatIndex Index2>
struct IsSameCharacter<
        TensorContravariantNaturalIndex<Index1>,
        TensorContravariantNaturalIndex<Index2>>
{
    static constexpr bool value = true;
};

} // namespace detail

template <class T1, class T2>
constexpr bool is_same_character_v = detail::IsSameCharacter<T1, T2>::value;

namespace detail {

template <class T1, class T2>
struct AreSameCharacters;

template <
        template <TensorNatIndex...>
        class T,
        TensorNatIndex HeadIndex1,
        TensorNatIndex... TailIndex1,
        TensorNatIndex HeadIndex2,
        TensorNatIndex... TailIndex2>
struct AreSameCharacters<T<HeadIndex1, TailIndex1...>, T<HeadIndex2, TailIndex2...>>
{
    static constexpr bool value = IsSameCharacter<HeadIndex1, HeadIndex2>::value
                                  && AreSameCharacters<T<TailIndex1...>, T<TailIndex2...>>::value;
};

} // namespace detail

template <class T1, class T2>
constexpr bool are_same_characters_v = detail::AreSameCharacters<T1, T2>::value;

namespace detail {

template <class T1, class T2>
struct AreDifferentCharacters;

template <template <TensorNatIndex...> class T>
struct AreDifferentCharacters<T<>, T<>>
{
    static constexpr bool value = true;
};

template <
        template <TensorNatIndex...>
        class T,
        TensorNatIndex HeadIndex1,
        TensorNatIndex... TailIndex1,
        TensorNatIndex HeadIndex2,
        TensorNatIndex... TailIndex2>
struct AreDifferentCharacters<T<HeadIndex1, TailIndex1...>, T<HeadIndex2, TailIndex2...>>
{
    static constexpr bool value
            = !IsSameCharacter<HeadIndex1, HeadIndex2>::value
              && AreDifferentCharacters<T<TailIndex1...>, T<TailIndex2...>>::value;
};

} // namespace detail

template <class T1, class T2>
constexpr bool are_different_characters_v = detail::AreDifferentCharacters<T1, T2>::value;

} // namespace tensor

} // namespace sil
