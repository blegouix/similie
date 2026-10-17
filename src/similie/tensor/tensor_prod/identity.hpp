// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/tensor/tensor_indices/identity_tensor.hpp>

namespace sil::tensor::detail {

template <class Index>
struct IdentityTensorProductMatcher
{
    template <class... Known, class... Tags>
    KOKKOS_FUNCTION static constexpr bool operator()(
            ddc::TypeSeq<Known...>,
            ddc::DiscreteElement<Tags...> element)
    {
        std::array<std::size_t, sizeof...(Known)> const ids {element.template uid<Known>()...};
        for (std::size_t i = 0; i < ids.size(); ++i)
            for (std::size_t j = i + 1; j < ids.size(); ++j)
                if (ids[i] != ids[j])
                    return false;
        return true;
    }
};

} // namespace sil::tensor::detail
