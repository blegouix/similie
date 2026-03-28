// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <similie/misc/specialization.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"

namespace sil {

namespace exterior {

namespace detail {

template <class T, class ElementType, class LayoutStridedPolicy>
struct FormWrapper;

template <misc::NotSpecialization<Chain> T, class ElementType, class LayoutStridedPolicy>
struct FormWrapper<T, ElementType, LayoutStridedPolicy>
{
    using type = Cosimplex<T, ElementType>;
};

template <misc::Specialization<Chain> T, class ElementType, class LayoutStridedPolicy>
struct FormWrapper<T, ElementType, LayoutStridedPolicy>
{
    using type = Cochain<T, ElementType, LayoutStridedPolicy>;
};

/*
template <misc::NotSpecialization<Chain> Head, class ElementType, class LayoutStridedPoliy >
FormWrapper(Head, ElementType) -> FormWrapper<Head, ElementType, LayoutStridedPolicy, Kokkos::DefaultHostExecutionSpace>;

template <misc::Specialization<Chain> Head, class ElementType, class LayoutStridedPolicy, class ExecSpace>
FormWrapper(Head, ElementType, Allocator) -> FormWrapper<Head, ElementType, LayoutStridedPolicy, ExecSpace>;
*/

} // namespace detail

// Usage should be avoided because CTAD cannot go through it
template <class T, class ElementType = double, class LayoutStridedPolicy = Kokkos::LayoutRight>
using Form = typename detail::FormWrapper<T, ElementType, LayoutStridedPolicy>::type;

template <class Tag, misc::Specialization<tensor::Tensor> TensorType>
struct FormComponent
{
    using tag = Tag;
    using tensor_type = TensorType;

    TensorType tensor;

    KOKKOS_FUNCTION explicit constexpr FormComponent(TensorType tensor_) noexcept : tensor(tensor_) {}
};

template <class Tag, misc::Specialization<tensor::Tensor> TensorType>
KOKKOS_FUNCTION constexpr FormComponent<Tag, TensorType> component(TensorType tensor) noexcept
{
    return FormComponent<Tag, TensorType>(tensor);
}

namespace detail {

template <class Tag, class... Components>
struct FormComponentByTag;

template <class Tag, class HeadComponent, class... TailComponents>
struct FormComponentByTag<Tag, HeadComponent, TailComponents...>
{
    using type = std::conditional_t<
            std::is_same_v<Tag, typename HeadComponent::tag>,
            HeadComponent,
            typename FormComponentByTag<Tag, TailComponents...>::type>;
};

template <class Tag>
struct FormComponentByTag<Tag>
{
    using type = void;
};

template <class... Components>
constexpr bool has_unique_component_tags_v
        = ((... && !std::is_same_v<typename Components::tag, typename Components::tag>) || true);

template <class HeadComponent, class... TailComponents>
constexpr bool has_unique_component_tags_v<HeadComponent, TailComponents...>
        = ((... && !std::is_same_v<typename HeadComponent::tag, typename TailComponents::tag>))
          && has_unique_component_tags_v<TailComponents...>;

} // namespace detail

template <class... Components>
class TensorForm
{
    static_assert(sizeof...(Components) > 0);
    static_assert(detail::has_unique_component_tags_v<Components...>);

    std::tuple<Components...> m_components;

public:
    using components_tuple_type = std::tuple<Components...>;

    constexpr explicit TensorForm(Components... components) noexcept
        : m_components(std::move(components)...)
    {
    }

    template <class Tag>
    KOKKOS_FUNCTION constexpr auto component() const noexcept
    {
        using component_t = typename detail::FormComponentByTag<Tag, Components...>::type;
        static_assert(!std::is_void_v<component_t>);
        return std::get<component_t>(m_components).tensor;
    }
};

template <class... Components>
constexpr TensorForm<Components...> make_tensor_form(Components... components) noexcept
{
    return TensorForm<Components...>(components...);
}

} // namespace exterior

} // namespace sil
