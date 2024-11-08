// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <fstream>

#include <ddc/ddc.hpp>

#include "csr_dynamic.hpp"
#include "tensor_impl.hpp"

namespace sil {

namespace csr {

namespace detail {

template <class Ids>
struct ArrayOfVectorsToArrayOfArrays;

template <std::size_t... I>
struct ArrayOfVectorsToArrayOfArrays<std::index_sequence<I...>>
{
    template <std::size_t N>
    static void run(
            std::array<std::array<std::size_t, N>, sizeof...(I)>& arr,
            std::array<std::vector<std::size_t>, sizeof...(I)> const& vec)
    {
        (std::copy_n(vec[I].begin(), N, arr[I].begin()), ...);
    }
};

} // namespace detail

// Only natural indexing supported
template <std::size_t N, class HeadTensorIndex, class... TailTensorIndex>
class Csr
{
private:
    ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> m_domain;
    std::array<std::size_t, HeadTensorIndex::mem_size() + 1> m_coalesc_idx;
    std::array<std::array<std::size_t, N>, sizeof...(TailTensorIndex)> m_idx;
    std::array<double, N> m_values;

public:
    constexpr Csr(
            ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain,
            std::array<std::size_t, HeadTensorIndex::mem_size() + 1> coalesc_idx,
            std::array<std::array<std::size_t, N>, sizeof...(TailTensorIndex)> idx,
            std::array<double, N> values)
        : m_domain(domain)
        , m_coalesc_idx(coalesc_idx)
        , m_idx(idx)
        , m_values(values)
    {
    }

    Csr(CsrDynamic<HeadTensorIndex, TailTensorIndex...> csr_dyn) : m_domain(csr_dyn.domain())
    {
        std::
                copy_n(csr_dyn.coalesc_idx().begin(),
                       HeadTensorIndex::mem_size() + 1,
                       m_coalesc_idx.begin());
        detail::ArrayOfVectorsToArrayOfArrays<
                std::make_index_sequence<sizeof...(TailTensorIndex)>>::run(m_idx, csr_dyn.idx());
        std::copy_n(csr_dyn.values().begin(), N, m_values.begin());
    }

    ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain()
    {
        return m_domain;
    }

    constexpr std::array<std::size_t, HeadTensorIndex::mem_size() + 1> coalesc_idx() const
    {
        return m_coalesc_idx;
    }

    constexpr std::array<std::array<std::size_t, N>, sizeof...(TailTensorIndex)> idx() const
    {
        return m_idx;
    }

    constexpr std::array<double, N> values() const
    {
        return m_values;
    }
};

/*
 Vector-Csr multiplication 
 */
template <std::size_t N, class HeadTensorIndex, class... TailTensorIndex>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<TailTensorIndex...>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<TailTensorIndex...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> prod,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadTensorIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> dense,
        Csr<N, HeadTensorIndex, TailTensorIndex...> csr)
{
    ddc::parallel_fill(prod, 0.);
    for (std::size_t i = 0; i < csr.coalesc_idx().size() - 1;
         ++i) { // TODO base on iterator ? Kokkosify ?
        double const dense_value = dense.mem(ddc::DiscreteElement<HeadTensorIndex>(i));
        std::size_t const j_begin = csr.coalesc_idx()[i];
        std::size_t const j_end = csr.coalesc_idx()[i + 1];
        Kokkos::parallel_for(
                "vector_dense_multiplication",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(j_begin, j_end),
                [&](const int j) {
                    prod(ddc::DiscreteElement<TailTensorIndex...>(csr.idx()[ddc::type_seq_rank_v<
                            TailTensorIndex,
                            ddc::detail::TypeSeq<TailTensorIndex...>>][j]...))
                            += dense_value * csr.values()[j];
                });
    }
    return prod;
}

/*
 Csr-dense multiplication 
 */
template <std::size_t N, class HeadTensorIndex, class... TailTensorIndex>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<HeadTensorIndex>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadTensorIndex>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> prod,
        Csr<N, HeadTensorIndex, TailTensorIndex...> csr,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<TailTensorIndex...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> dense)
{
    ddc::parallel_fill(prod, 0.);
    for (std::size_t i = 0; i < csr.coalesc_idx().size() - 1;
         ++i) { // TODO base on iterator ? Kokkosify ?
        std::size_t const j_begin = csr.coalesc_idx()[i];
        std::size_t const j_end = csr.coalesc_idx()[i + 1];
        Kokkos::parallel_reduce(
                "csr_dense_multiplication",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(j_begin, j_end),
                [&](const int j, double& lsum) {
                    double const dense_value = dense(
                            ddc::DiscreteElement<TailTensorIndex...>(csr.idx()[ddc::type_seq_rank_v<
                                    TailTensorIndex,
                                    ddc::detail::TypeSeq<TailTensorIndex...>>][j]...));
                    lsum += dense_value * csr.values()[j];
                },
                prod.mem(ddc::DiscreteElement<HeadTensorIndex>(i)));
    }
    return prod;
}

template <std::size_t N, class... TensorIndex>
std::ostream& operator<<(std::ostream& os, Csr<N, TensorIndex...> const& csr)
{
    os << "----------\n";
    for (std::size_t i = 0; i < csr.coalesc_idx().size(); ++i) {
        os << csr.coalesc_idx()[i] << " ";
    }
    os << "\n";
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < sizeof...(TensorIndex) - 1; ++j) {
            os << csr.idx()[j][i] << " ";
        }
        os << csr.values()[i];
        os << "\n";
    }
    return os;
}

} // namespace csr

} // namespace sil
