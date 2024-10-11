// SPDX-License-Identifier: GPL-3.0

#include <cmath>

#include <gtest/gtest.h>

#include "young_tableau.hpp"

using TableauSeq
        = sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2, 4>>;

TEST(YoungTableau, Init)
{
    sil::young_tableau::YoungTableau<TableauSeq> young_tableau;
}
