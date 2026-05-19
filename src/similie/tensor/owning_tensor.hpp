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

    KOKKOS_FUNCTION void rebind_to_owned_storage(DomainType const& domain)
    {
        static_cast<tensor_type&>(*this) = tensor_type(
                ddc::ChunkSpan<
                        ElementType,
                        DomainType,
                        LayoutStridedPolicy,
                        MemorySpace>(m_storage.data(), domain));
    }

public:
    KOKKOS_FUNCTION explicit OwningTensor(tensor_type tensor, Storage&& storage)
        : tensor_type(tensor)
        , m_storage(std::move(storage))
    {
        rebind_to_owned_storage(tensor.domain());
    }

    KOKKOS_FUNCTION OwningTensor(OwningTensor const& other)
        : tensor_type(static_cast<tensor_type const&>(other))
        , m_storage(other.m_storage)
    {
        rebind_to_owned_storage(other.domain());
    }

    KOKKOS_FUNCTION OwningTensor(OwningTensor&& other)
        : tensor_type(static_cast<tensor_type&&>(other))
        , m_storage(std::move(other.m_storage))
    {
        rebind_to_owned_storage(other.domain());
    }

    KOKKOS_FUNCTION OwningTensor& operator=(OwningTensor const& other)
    {
        m_storage = other.m_storage;
        rebind_to_owned_storage(other.domain());
        return *this;
    }

    KOKKOS_FUNCTION OwningTensor& operator=(OwningTensor&& other)
    {
        m_storage = std::move(other.m_storage);
        rebind_to_owned_storage(other.domain());
        return *this;
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
