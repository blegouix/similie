// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

namespace detail {

template <class ChunkDst, class ChunkSrc>
auto unsecure_parallel_deepcopy(ChunkDst&& dst, ChunkSrc&& src)
{
    static_assert(ddc::is_borrowed_chunk_v<ChunkDst>);
    static_assert(ddc::is_borrowed_chunk_v<ChunkSrc>);
    static_assert(
            std::is_assignable_v<
                    ddc::chunk_reference_t<ChunkDst>,
                    ddc::chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    Kokkos::deep_copy(dst.allocation_kokkos_view(), src.allocation_kokkos_view());
    return dst.span_view();
}

template <class ExecSpace, class ChunkDst, class ChunkSrc>
auto unsecure_parallel_deepcopy(ExecSpace const& execution_space, ChunkDst&& dst, ChunkSrc&& src)
{
    static_assert(ddc::is_borrowed_chunk_v<ChunkDst>);
    static_assert(ddc::is_borrowed_chunk_v<ChunkSrc>);
    static_assert(
            std::is_assignable_v<
                    ddc::chunk_reference_t<ChunkDst>,
                    ddc::chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    Kokkos::deep_copy(execution_space, dst.allocation_kokkos_view(), src.allocation_kokkos_view());
    return dst.span_view();
}

} // namespace detail

} // namespace misc

} // namespace sil
