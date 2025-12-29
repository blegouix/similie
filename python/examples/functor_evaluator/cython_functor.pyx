# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

from libc.stdlib cimport malloc, free

cdef extern from "cython_functor_api.hpp":
    cdef struct CythonFunctorHandle:
        void* ptr

cdef extern from *:
    """
    double evaluate_with_cpp(CythonFunctorHandle* handle, int x, int y, int z);
    """
    double evaluate_with_cpp(CythonFunctorHandle* handle, int x, int y, int z)


cdef class CythonFunctor:
    cdef double factor

    def __cinit__(self, double factor):
        self.factor = factor

    cpdef double __call__(self, int x, int y, int z):
        return self.factor * (100.0 * z + 10.0 * y + x)


cdef CythonFunctorHandle* allocate_functor(double factor):
    cdef CythonFunctorHandle* handle = <CythonFunctorHandle*>malloc(sizeof(CythonFunctorHandle))
    handle.ptr = <void*>CythonFunctor(factor)
    return handle


cpdef CythonFunctorHandle* cython_functor_create(double factor):
    return allocate_functor(factor)


cpdef void cython_functor_destroy(CythonFunctorHandle* handle):
    if handle is not NULL and handle.ptr is not NULL:
        del <CythonFunctor>handle.ptr
        free(handle)


cpdef double evaluate_from_cpp(CythonFunctorHandle* handle, int x, int y, int z):
    return evaluate_with_cpp(handle, x, y, z)


cdef public double cython_functor_call(CythonFunctorHandle* handle, int x, int y, int z):
    if handle is NULL or handle.ptr is NULL:
        raise ValueError("Invalid CythonFunctorHandle")
    return (<CythonFunctor>handle.ptr)(x, y, z)
