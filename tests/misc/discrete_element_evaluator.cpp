// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <memory>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <similie/misc/discrete_element_evaluator.hpp>

struct DDimX
{
};

struct DDimY
{
};

struct DDimZ
{
};

TEST(FunctorEvaluatorByOrderedIndices, ForwardsIndicesToFunctor)
{
    struct ExternalFunctor {
        double factor;

        double operator()(int x, int y, int z) const
        {
            return factor * (100.0 * z + 10.0 * y + x);
        }
    };

    using domain_t = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;

    sil::misc::FunctorEvaluatorByOrderedIndices<domain_t, ExternalFunctor> evaluator(
            ExternalFunctor {0.5});

    ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem(52, 34, 1);

    EXPECT_DOUBLE_EQ(evaluator(elem), 0.5 * (100.0 + 340.0 + 52.0));
}

TEST(FunctorEvaluatorByOrderedIndices, SupportsMoveOnlyFunctor)
{
    struct MoveOnlyFunctor {
        std::unique_ptr<int> weight;

        double operator()(int x) const
        {
            return static_cast<double>(*weight * x);
        }
    };

    using domain_t = ddc::DiscreteDomain<DDimX>;

    sil::misc::FunctorEvaluatorByOrderedIndices<domain_t, MoveOnlyFunctor> evaluator(
            MoveOnlyFunctor {std::make_unique<int>(3)});

    ddc::DiscreteElement<DDimX> elem(7);

    EXPECT_DOUBLE_EQ(evaluator(elem), 21.);
}
