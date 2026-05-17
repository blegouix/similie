// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "tensor_impl.hpp"

namespace sil::tensor {

template <class TensorType, class Storage>
class OwningTensor;

template <
        class ElementType,
        class DomainType,
        class LayoutStridedPolicy,
        class MemorySpace,
        class Storage>
class OwningTensor<Tensor<ElementType, DomainType, LayoutStridedPolicy, MemorySpace>, Storage>
    : public Tensor<ElementType, DomainType, LayoutStridedPolicy, MemorySpace>
{
    using tensor_type = Tensor<ElementType, DomainType, LayoutStridedPolicy, MemorySpace>;

    Storage m_storage;

public:
    KOKKOS_FUNCTION explicit OwningTensor(tensor_type tensor, Storage&& storage)
        : tensor_type(tensor)
        , m_storage(std::move(storage))
    {
    }

    KOKKOS_FUNCTION Storage& storage() noexcept
    {
        return m_storage;
    }

    KOKKOS_FUNCTION Storage const& storage() const noexcept
    {
        return m_storage;
    }
};

template <class TensorType, class Storage>
OwningTensor(TensorType, Storage&&) -> OwningTensor<TensorType, Storage>;

} // namespace sil::tensor
