// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <fstream>

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace csr {

// Only natural indexing supported
template <class HeadTensorIndex, class... TailTensorIndex>
class Csr
{
private:
    ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> m_domain;
    std::vector<std::size_t> m_coalesc_idx;
    std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> m_idx;
    std::vector<double> m_values;

public:
    Csr(ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain)
        : m_domain(domain)
        , m_coalesc_idx({0})
        , m_idx()
        , m_values()
    {
    }

    Csr(ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain,
        std::vector<std::size_t> coalesc_idx,
        std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> idx,
        std::vector<double> values)
        : m_domain(domain)
        , m_coalesc_idx(coalesc_idx)
        , m_idx(idx)
        , m_values(values)
    {
    }

    ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain()
    {
        return m_domain;
    }

    std::vector<std::size_t> coalesc_idx()
    {
        return m_coalesc_idx;
    }

    std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> idx()
    {
        return m_idx;
    }

    std::vector<double> values()
    {
        return m_values;
    }

    void push_back(sil::tensor::Tensor<
                   double,
                   ddc::DiscreteDomain<TailTensorIndex...>,
                   std::experimental::layout_right,
                   Kokkos::DefaultHostExecutionSpace::memory_space> dense)
    {
        m_coalesc_idx.push_back(m_coalesc_idx.back());
        ddc::for_each(dense.domain(), [&](ddc::DiscreteElement<TailTensorIndex...> elem) {
            if (dense(elem) != 0) {
                m_coalesc_idx.back() += 1;
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
    Csr<HeadTensorIndex, TailTensorIndex...> get(ddc::DiscreteElement<HeadTensorIndex> id) const
    {
        const std::size_t id_begin = m_coalesc_idx[id.uid()];
        const std::size_t id_end = m_coalesc_idx[id.uid() + 1];
        std::vector<std::size_t> new_coalesc_idx {0, id_end - id_begin};
        std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> new_idx;
        ((new_idx[ddc::type_seq_rank_v<TailTensorIndex, ddc::detail::TypeSeq<TailTensorIndex...>>]
          = std::vector<std::size_t>(
                  m_idx[ddc::type_seq_rank_v<
                                TailTensorIndex,
                                ddc::detail::TypeSeq<TailTensorIndex...>>]
                                  .begin()
                          + id_begin,
                  m_idx[ddc::type_seq_rank_v<
                                TailTensorIndex,
                                ddc::detail::TypeSeq<TailTensorIndex...>>]
                                  .begin()
                          + id_begin + id_end)),
         ...);
        std::vector<double>
                new_values(m_values.begin() + id_begin, m_values.begin() + id_begin + id_end);

        return Csr<HeadTensorIndex, TailTensorIndex...>(
                ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...>(m_domain),
                new_coalesc_idx,
                new_idx,
                new_values);
    }

    void write(const std::string& filename, std::string irrep_tag)
    {
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        file << irrep_tag << "\n";
        file
                .write(reinterpret_cast<const char*>(m_values.data()),
                       m_values.size() * sizeof(double));
        file << "\n\n";
        file.close();
        if (!file.good()) {
            std::cerr << "Error occurred while writing to file: " << filename << std::endl;
        } else {
            std::cout << "File written successfully: " << filename << std::endl;
        }
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
    for (std::size_t i = 0; i < csr.coalesc_idx().size() - 1;
         ++i) { // TODO base on iterator ? Kokkosify ?
        double const dense_value = dense(ddc::DiscreteElement<HeadTensorIndex>(i));
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
                prod(ddc::DiscreteElement<HeadTensorIndex>(i)));
    }
    return prod;
}



// Convert Csr to dense tensor
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
                          ddc::DiscreteElement<TailId...>(csr.idx()[ddc::type_seq_rank_v<
                                  TailId,
                                  ddc::detail::TypeSeq<TailId...>>][j]...))
                            = csr.values()[j];
                });
    }
    return dense;
}

} // namespace csr

} // namespace sil
