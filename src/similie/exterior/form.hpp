// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <array>
#include <cassert>
#include <initializer_list>
#include <utility>

#include <similie/tensor/owning_tensor.hpp>

namespace sil::exterior {

namespace detail {
template <class T>
struct FormArrayStorage : std::false_type
{
};
template <class T, std::size_t N>
struct FormArrayStorage<std::array<T, N>> : std::true_type
{
};
} // namespace detail

/** Differential-form coefficients, pointwise or sampled over a spatial domain.
 * \important This documentation is fully AI-generated.
 */
template <class Coefficients>
class Form : public Coefficients
{
    static_assert(
            misc::Specialization<Coefficients, tensor::Tensor>
            || misc::Specialization<Coefficients, tensor::OwningTensor>
            || detail::FormArrayStorage<Coefficients>::value);

public:
    using coefficients_type = Coefficients;

    KOKKOS_DEFAULTED_FUNCTION constexpr Form() = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr Form(Form const&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr Form(Form&&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr Form& operator=(Form const&) = default;
    KOKKOS_DEFAULTED_FUNCTION constexpr Form& operator=(Form&&) = default;

    KOKKOS_FUNCTION constexpr explicit Form(Coefficients const& coefficients)
        : Coefficients(coefficients)
    {
    }

    KOKKOS_FUNCTION constexpr Form(std::initializer_list<typename Coefficients::value_type> values)
        requires(detail::FormArrayStorage<Coefficients>::value)
        : Coefficients {}
    {
        if (values.size() > this->size())
            Kokkos::abort("Too many differential-form coefficients");
        std::size_t i = 0;
        for (typename Coefficients::value_type const& value : values)
            Coefficients::operator[](i++) = value;
    }

    KOKKOS_FUNCTION constexpr decltype(auto) coefficients() const
    {
        if constexpr (misc::Specialization<Coefficients, tensor::OwningTensor>)
            return static_cast<typename Coefficients::tensor_type const&>(*this);
        else
            return static_cast<Coefficients const&>(*this);
    }
    KOKKOS_FUNCTION constexpr decltype(auto) coefficients()
    {
        if constexpr (misc::Specialization<Coefficients, tensor::OwningTensor>)
            return static_cast<typename Coefficients::tensor_type&>(*this);
        else
            return static_cast<Coefficients&>(*this);
    }

    KOKKOS_FUNCTION Form& operator+=(Form const& other)
        requires requires(Coefficients& left, Coefficients const& right) { left += right; }
    {
        coefficients() += other.coefficients();
        return *this;
    }

    template <class Scalar>
    KOKKOS_FUNCTION Form& operator*=(Scalar scalar)
        requires requires(Coefficients& coefficients) { coefficients *= scalar; }
    {
        coefficients() *= scalar;
        return *this;
    }

    template <class Element>
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](Element element) const
    {
        if constexpr (!detail::FormArrayStorage<Coefficients>::value)
            return ::sil::exterior::Form(Coefficients::operator[](element));
        else
            return Coefficients::operator[](element);
    }

    template <class Element>
    KOKKOS_FUNCTION constexpr decltype(auto) operator[](Element element)
    {
        if constexpr (!detail::FormArrayStorage<Coefficients>::value)
            return ::sil::exterior::Form(Coefficients::operator[](element));
        else
            return Coefficients::operator[](element);
    }
};

// A scalar physical field is a zero-form. Arithmetic reads its pointwise
// coefficient; constructing one from an unclassified number remains explicit.
template <>
class Form<double>
{
    double m_value = 0.0;

public:
    using coefficients_type = double;
    KOKKOS_DEFAULTED_FUNCTION constexpr Form() = default;
    KOKKOS_FUNCTION constexpr explicit Form(double value) : m_value(value) {}
    KOKKOS_FUNCTION constexpr operator double() const { return m_value; }
    KOKKOS_FUNCTION constexpr double value() const { return m_value; }
    KOKKOS_FUNCTION constexpr double const& coefficients() const { return m_value; }
    KOKKOS_FUNCTION constexpr double& coefficients() { return m_value; }
    KOKKOS_FUNCTION constexpr Form& operator=(double value) { m_value = value; return *this; }
    KOKKOS_FUNCTION constexpr Form& operator+=(double value) { m_value += value; return *this; }
    KOKKOS_FUNCTION constexpr Form& operator-=(double value) { m_value -= value; return *this; }
    KOKKOS_FUNCTION constexpr Form& operator*=(double value) { m_value *= value; return *this; }
    KOKKOS_FUNCTION constexpr Form& operator/=(double value) { m_value /= value; return *this; }
};

template <class Coefficients>
Form(Coefficients const&) -> Form<Coefficients>;

template <class T>
concept FormType = misc::Specialization<std::remove_cvref_t<T>, Form>;

template <class T>
concept TensorFormType = FormType<T> && requires { typename T::indices_domain_t; };

namespace detail {
template <class Domain>
struct FormDegree;
template <class... Index>
struct FormDegree<ddc::DiscreteDomain<Index...>>
    : std::integral_constant<std::size_t, (Index::rank() + ... + 0)> {};
} // namespace detail

template <TensorFormType T>
inline constexpr std::size_t form_degree_v = detail::FormDegree<typename T::indices_domain_t>::value;

} // namespace sil::exterior
