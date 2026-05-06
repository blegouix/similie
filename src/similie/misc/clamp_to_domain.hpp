// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <Kokkos_MinMax.hpp>

namespace sil {

namespace misc {

template <class BatchDomain, class Elem>
KOKKOS_FUNCTION Elem clamp_to_domain(BatchDomain const& batch_domain, Elem elem)
{
    auto const front = batch_domain.front();
    auto const back = batch_domain.back();
    for (std::size_t i = 0; i < Elem::size(); ++i) {
        ddc::detail::array(elem)[i] = Kokkos::
                min(Kokkos::max(ddc::detail::array(elem)[i], ddc::detail::array(front)[i]),
                    ddc::detail::array(back)[i]);
    }
    return elem;
}

} // namespace misc

} // namespace sil
