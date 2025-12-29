# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

cdef extern from *:
    """
    #include <stdexcept>
    #include <Python.h>
    #include <ddc/ddc.hpp>
    #include <similie/misc/discrete_element_evaluator.hpp>

    struct DDimX {};
    struct DDimY {};
    struct DDimZ {};

    using domain_t = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;

    struct PyFunctorAdapter {
        PyObject* obj;

        PyFunctorAdapter(PyObject* o)
            : obj(o)
        {
            Py_XINCREF(obj);
        }

        PyFunctorAdapter(PyFunctorAdapter const& other)
            : obj(other.obj)
        {
            Py_XINCREF(obj);
        }

        ~PyFunctorAdapter()
        {
            Py_XDECREF(obj);
        }

        double operator()(int x, int y, int z) const
        {
            PyObject* args = Py_BuildValue("(iii)", x, y, z);
            if (args == nullptr) {
                throw std::runtime_error("Failed to build args");
            }
            PyObject* result = PyObject_Call(obj, args, nullptr);
            Py_DECREF(args);
            if (result == nullptr) {
                throw std::runtime_error("Python functor call failed");
            }
            double value = PyFloat_AsDouble(result);
            Py_DECREF(result);
            if (PyErr_Occurred()) {
                throw std::runtime_error("Python functor did not return a float-convertible value");
            }
            return value;
        }
    };

    inline double evaluate_with_cpp(PyObject* functor, int x, int y, int z)
    {
        PyFunctorAdapter adapter(functor);
        sil::misc::FunctorEvaluatorByOrderedIndices<domain_t, PyFunctorAdapter> evaluator(adapter);
        ddc::DiscreteElement<DDimX, DDimY, DDimZ> elem(x, y, z);
        return evaluator(elem);
    }
    """
    double evaluate_with_cpp(object functor, int x, int y, int z)


cdef class CythonFunctor:
    cdef double factor

    def __cinit__(self, double factor):
        self.factor = factor

    cpdef double __call__(self, int x, int y, int z):
        return self.factor * (100.0 * z + 10.0 * y + x)


def evaluate_functor(double factor, int x, int y, int z):
    """
    Build a CythonFunctor and evaluate it through the C++ FunctorEvaluatorByOrderedIndices.
    """
    functor = CythonFunctor(factor)
    return evaluate_with_cpp(functor, x, y, z)
