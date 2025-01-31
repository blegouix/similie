// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace misc {

template <class TagSeqA, class TagSeqB>
using type_seq_intersect_t
        = ddc::type_seq_remove_t<TagSeqA, ddc::type_seq_remove_t<TagSeqA, TagSeqB>>;

} // namespace misc

} // namespace sil
