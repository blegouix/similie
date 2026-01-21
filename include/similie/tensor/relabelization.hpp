// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

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
        template <class...> class IndexToRelabelizeType,
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
    static constexpr auto run(ddc::DiscreteElement<DDim...> elem)
    {
        return ddc::DiscreteElement<
                typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::type...>(
                elem.template uid<DDim>()...);
    }

    template <class... DDim>
    static constexpr auto run(ddc::DiscreteVector<DDim...> vect)
    {
        return ddc::DiscreteVector<
                typename detail::RelabelizeIndex<DDim, OldIndex, NewIndex>::type...>(
                static_cast<std::size_t>(vect.template get<DDim>())...);
    }

    template <class... DDim>
    static constexpr auto run(ddc::DiscreteDomain<DDim...> dom)
    {
        return relabelize_index_in_t<ddc::DiscreteDomain<DDim...>, OldIndex, NewIndex>(
                relabelize_index_in<OldIndex, NewIndex>(dom.front()),
                relabelize_index_in<OldIndex, NewIndex>(dom.extents()));
    }
};

} // namespace detail

template <class OldIndex, class NewIndex, class T>
constexpr relabelize_index_in_t<T, OldIndex, NewIndex> relabelize_index_in(T t)
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
constexpr relabelize_index_of_t<
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
    static constexpr auto run(ddc::DiscreteElement<DDim...> elem)
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
    static constexpr auto run(ddc::DiscreteVector<DDim...> vect)
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
    static constexpr auto run(ddc::DiscreteDomain<DDim...> dom)
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
constexpr relabelize_indices_in_t<T, OldIndices, NewIndices> relabelize_indices_in(T t)
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
constexpr auto RelabelizeIndicesOf(
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
constexpr relabelize_indices_of_t<Tensor, OldIndices, NewIndices> relabelize_indices_of(
        Tensor tensor)
{
    static_assert(ddc::type_seq_size_v<OldIndices> == ddc::type_seq_size_v<NewIndices>);
    return detail::RelabelizeIndicesOf<OldIndices, NewIndices, 0>(tensor);
}

} // namespace tensor

} // namespace sil
