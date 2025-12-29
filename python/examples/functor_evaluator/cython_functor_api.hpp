// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

struct CythonFunctorHandle
{
    void* ptr;
};

extern "C" double cython_functor_call(CythonFunctorHandle* handle, int x, int y, int z);
