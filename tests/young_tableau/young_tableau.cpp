// SPDX-License-Identifier: GPL-3.0

#include <cmath>

#include <gtest/gtest.h>

#include "young_tableau.hpp"

TEST(YoungTableau, IrrepDim1_2)
{
    sil::young_tableau::
            YoungTableau<4, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2>>>
                    young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 10);
}

TEST(YoungTableau, IrrepDim1l2)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 6);
}

TEST(YoungTableau, IrrepDim1_2_3)
{
    sil::young_tableau::
            YoungTableau<4, sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2, 3>>>
                    young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}

TEST(YoungTableau, IrrepDim1l2l3)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<
                    std::index_sequence<1>,
                    std::index_sequence<2>,
                    std::index_sequence<3>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 4);
}

TEST(YoungTableau, IrrepDim1_2l3)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 2>, std::index_sequence<3>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}

TEST(YoungTableau, IrrepDim1_3l2)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}

TEST(YoungTableau, IrrepDim1l3_2l4)
{
    sil::young_tableau::YoungTableau<
            4,
            sil::young_tableau::
                    YoungTableauSeq<std::index_sequence<1, 3>, std::index_sequence<2, 4>>>
            young_tableau;

    EXPECT_EQ(young_tableau.irrep_dim(), 20);
}
