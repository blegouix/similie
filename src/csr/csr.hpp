// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "tensor.hpp"

namespace sil {

namespace csr {

// Only natural indexing supported
template <class HeadTensorIndex, class... TailTensorIndex>
class Csr
{
private:
    ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> m_domain;
    std::array<std::size_t, HeadTensorIndex::size() + 1> m_coalesc_idx;
    std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> m_idx;
    std::vector<double> m_values;

public:
    Csr(ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain)
        : m_domain(domain)
        , (m_idx[ddc::type_seq_rank_v<TailTensorIndex, ddc::detail::TypeSeq<TailTensorIndex...>>](
                   sizeof...(TailTensorIndex)),
           ...)
    {
        m_coalesc_idx[0] = 1;
    }

    // TODO necessary ?
    Csr(ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain,
        std::array<std::size_t, HeadTensorIndex::size() + 1> coalesc_idx,
        std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> idx,
        std::vector<double> values)
        : m_domain(domain)
        , m_coalesc_idx(coalesc_idx)
        , m_idx(idx)
        , m_values(values)
    {
    }

    static std::array<std::size_t, HeadTensorIndex::size() + 1> coalesc_idx()
    {
        return m_coalesc_idx;
    }

    static std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> idx()
    {
        return m_idx;
    }

    static std::vector<double> values()
    {
        return m_values;
    }

    void push_back(sil::tensor::Tensor<
                   double,
                   ddc::DiscreteDomain<TailTensorIndex...>,
                   std::experimental::layout_right,
                   Kokkos::DefaultHostExecutionSpace::memory_space> dense)
    {
        m_coalesc_idx[m_values.size() + 1] = m_coalesc_idx[m_values.size()];
        ddc::for_each(dense.domain(), [&](ddc::DiscreteElement<TailTensorIndex...> elem) {
            if (dense(elem) != 0) {
                m_coalesc_idx[m_n_nonzeros + 1] += 1;
                (m_idx[ddc::type_seq_rank_v<
                               TailTensorIndex,
                               ddc::detail::TypeSeq<TailTensorIndex...>>]
                         .push_back(elem.template uid<TailTensorIndex>()),
                 ...);
                m_values.push_back(dense(elem));
            }
        });
    }

    // Returns a slice orthogonal to first index
    template <class Id>
    Csr<sil::tensor::TensorNaturalIndex<Id>, TailTensorIndex...> get() const
    {
        std::array<std::size_t, 2> new_coalesc_idx {
                0,
                m_coalesc_idx[HeadTensorIndex::access_id<Id>() + 1]
                        - m_coalesc_idx[HeadTensorIndex::access_id<Id>()]};
        std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> new_idx;
        (new_idx[ddc::type_seq_rank_v<TailTensorIndex, ddc::detail::TypeSeq<TailTensorIndex...>>](
                 m_idx[ddc::type_seq_rank_v<
                               TailTensorIndex,
                               ddc::detail::TypeSeq<TailTensorIndex...>>]
                                 .begin()
                         + new_coalesc_idx[HeadTensorIndex::access_id<Id>()],
                 m_idx[ddc::type_seq_rank_v<
                               TailTensorIndex,
                               ddc::detail::TypeSeq<TailTensorIndex...>>]
                                 .begin()
                         + m_coalesc_idx[HeadTensorIndex::access_id<Id>() + 1]),
         ...);
        std::vector<double> new_values(
                m_values.begin() + m_coalesc_idx[HeadTensorIndex::access_id<Id>()],
                m_values.begin() + m_coalesc_idx[HeadTensorIndex::access_id<Id>() + 1]);

        return Csr<sil::tensor::TensorNaturalIndex<Id>, TailTensorIndex...>
                slice(ddc::DiscreteDomain<sil::tensor::TensorNaturalIndex<Id>, TailTensorIndex...>(
                              m_domain),
                      new_coalesc_idx,
                      new_idx,
                      new_values);
    }

    // Print method for debugging
    void print() const
    {
        std::cout << "Dimensions: ";
        for (int dim : m_ndims)
            std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "Coalescent indices: ";
        for (int idx : m_coalesc_idx)
            std::cout << idx << " ";
        std::cout << std::endl;

        std::cout << "Indices:\n";
        for (const auto& row : m_idx) {
            for (int idx : row)
                std::cout << idx << " ";
            std::cout << std::endl;
        }

        std::cout << "Values: ";
        for (double val : m_values)
            std::cout << val << " ";
        std::cout << std::endl;
    }
};

/*
 Vector-Csr multiplication 
 */
template <class HeadTensorIndex, class... TailTensorIndex>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<TailTensorIndex...>,
        std::experimental::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<TailTensorIndex...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> prod,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadTensorIndex>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> dense,
        Csr<HeadTensorIndex, TailTensorIndex...> csr)
{
    ddc::parallel_fill(prod, 0.);
    for (std::size_t i = 0; i < csr.coalesc_idx().size();
         ++i) { // TODO base on iterator ? Kokkosify ?
        double const dense_value = dense(ddc::DiscreteElement<HeadTensorIndex>(i));
        std::size_t const j_begin = csr.coalesc_idx()[i];
        std::size_t const j_end = csr.coalesc_idx()[i + 1];
        Kokkos::parallel_for(
                "vector_dense_multiplication",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(j_begin, j_end),
                [&](const int j) {
                    prod(ddc::DiscreteElement<TailTensorIndex...>(
                            csr.idx()[ddc::type_seq_rank_v<
                                    TailTensorIndex,
                                    ddc::detail::TypeSeq<TailTensorIndex...>>][j],
                            ...))
                            += dense_value * csr.values[j];
                });
    }
    return prod;
}

/*
 Csr-vector multiplication 
 */
template <class HeadTensorIndex, class... TailTensorIndex>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<HeadTensorIndex>,
        std::experimental::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
tensor_prod(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadTensorIndex>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> prod,
        Csr<HeadTensorIndex, TailTensorIndex...> csr,
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<TailTensorIndex...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> dense)
{
    ddc::parallel_fill(prod, 0.);
    for (std::size_t i = 0; i < csr.coalesc_idx().size();
         ++i) { // TODO base on iterator ? Kokkosify ?
        std::size_t const j_begin = csr.coalesc_idx()[i];
        std::size_t const j_end = csr.coalesc_idx()[i + 1];
        Kokkos::parallel_reduce(
                "dense_vector_multiplication",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(j_begin, j_end),
                [&](const int j, double& lsum) {
                    double const dense_value = dense(ddc::DiscreteElement<TailTensorIndex...>(
                            csr.idx()[ddc::type_seq_rank_v<
                                    TailTensorIndex,
                                    ddc::detail::TypeSeq<TailTensorIndex...>>][j],
                            ...));
                    lsum += dense_value * csr.values[j];
                },
                prod(ddc::DiscreteElement<HeadTensorIndex>(i)));
    }
    return prod;
}



// Convert CSR to dense matrix
template <class HeadId, class... TailId>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<HeadId, TailId...>,
        std::experimental::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
csr2dense(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadId, TailId...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> dense,
        Csr<HeadId, TailId...> csr)
{
    ddc::parallel_fill(dense, 0.);
    for (std::size_t i = 0; i < csr.coalesc_idx().size() - 1; ++i) {
        std::size_t const j_begin = csr.coalesc_idx()[i];
        std::size_t const j_end = csr.coalesc_idx()[i + 1];
        Kokkos::parallel_for(
                "csr2dense",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(j_begin, j_end),
                [&](const int j) {
                    dense(ddc::DiscreteElement<HeadId>(i),
                          ddc::DiscreteElement<TailTensorIndex...>(
                                  csr.idx()[ddc::type_seq_rank_v<
                                          TailTensorIndex,
                                          ddc::detail::TypeSeq<TailTensorIndex...>>][j],
                                  ...))
                            = csr.m_values[j];
                });
    }
    return dense;
}

} // namespace csr

} // namespace sil
