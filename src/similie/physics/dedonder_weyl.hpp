// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

namespace similie::physics::dedonder_weyl {

template <class StrongFormOperator>
struct StationaryStrongFormulation
{
    StrongFormOperator strong_form_operator;
};

} // namespace similie::physics::dedonder_weyl
