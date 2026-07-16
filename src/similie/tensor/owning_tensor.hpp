// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

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
    using Base = Tensor<ElementType, DomainType, LayoutStridedPolicy, MemorySpace>;

    Storage m_storage;

    KOKKOS_FUNCTION void rebind_to_owned_storage(DomainType const& domain)
    {
        static_cast<Base&>(*this)
                = Base(ddc::ChunkSpan<
                        ElementType,
                        DomainType,
                        LayoutStridedPolicy,
                        MemorySpace>(m_storage.data(), domain));
    }

public:
    KOKKOS_FUNCTION explicit OwningTensor(Base tensor, Storage&& storage)
        : Base(tensor)
        , m_storage(std::move(storage))
    {
        rebind_to_owned_storage(tensor.domain());
    }

    KOKKOS_FUNCTION OwningTensor(OwningTensor const& other)
        : Base(static_cast<Base const&>(other))
        , m_storage(other.m_storage)
    {
        rebind_to_owned_storage(other.domain());
    }

    KOKKOS_FUNCTION OwningTensor(OwningTensor&& other)
        : Base(static_cast<Base&&>(other))
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
