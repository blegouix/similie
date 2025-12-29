// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <similie/misc/discrete_element_evaluator.hpp>

#include "cython_functor_api.hpp"

namespace {
struct DDimX
{
};

struct DDimY
{
};

struct DDimZ
{
};

using domain_t = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;

struct CythonFunctorWrapper
{
    CythonFunctorHandle* handle;

    double operator()(int x, int y, int z) const
    {
        return cython_functor_call(handle, x, y, z);
    }
};
} // namespace

extern "C" double evaluate_with_cpp(CythonFunctorHandle* handle, int x, int y, int z)
{
    CythonFunctorWrapper wrapper {handle};
    sil::misc::FunctorEvaluatorByOrderedIndices<domain_t, CythonFunctorWrapper> evaluator(
            wrapper);
    ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem(x, y, z);
    return evaluator(elem);
}
