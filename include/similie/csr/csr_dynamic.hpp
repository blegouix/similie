// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <fstream>

#include <ddc/ddc.hpp>

#include <similie/tensor/tensor_impl.hpp>

namespace sil {

namespace csr {

// Only natural indexing supported
template <tensor::TensorIndex HeadTensorIndex, tensor::TensorNatIndex... TailTensorIndex>
class CsrDynamic
{
private:
    ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> m_domain;
    std::vector<std::size_t> m_coalesc_idx;
    std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> m_idx;
    std::vector<double> m_values;

public:
    CsrDynamic(ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain)
        : m_domain(domain)
        , m_coalesc_idx({0})
        , m_idx()
        , m_values()
    {
    }

    CsrDynamic(
            ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...> domain,
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

    constexpr std::vector<std::size_t> coalesc_idx() const
    {
        return m_coalesc_idx;
    }

    constexpr std::array<std::vector<std::size_t>, sizeof...(TailTensorIndex)> idx() const
    {
        return m_idx;
    }

    constexpr std::vector<double> values() const
    {
        return m_values;
    }

    void push_back(sil::tensor::Tensor<
                   double,
                   ddc::DiscreteDomain<TailTensorIndex...>,
                   Kokkos::layout_right,
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
    CsrDynamic<HeadTensorIndex, TailTensorIndex...> get(
            ddc::DiscreteElement<HeadTensorIndex> id) const
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

        return CsrDynamic<HeadTensorIndex, TailTensorIndex...>(
                ddc::DiscreteDomain<HeadTensorIndex, TailTensorIndex...>(m_domain),
                new_coalesc_idx,
                new_idx,
                new_values);
    }

    void write(std::ofstream& file)
    {
        file
                .write(reinterpret_cast<const char*>(coalesc_idx().data()),
                       coalesc_idx().size() * sizeof(std::size_t));
        file << "break\n";
        for (std::size_t i = 0; i < sizeof...(TailTensorIndex); ++i) {
            file
                    .write(reinterpret_cast<const char*>(idx()[i].data()),
                           idx()[i].size() * sizeof(std::size_t));
            file << "break\n";
        }
        file
                .write(reinterpret_cast<const char*>(values().data()),
                       m_values.size() * sizeof(double));
        file << "break\n";
    }
};

// Convert Csr to dense tensor
template <tensor::TensorIndex HeadId, tensor::TensorNatIndex... TailId>
sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<HeadId, TailId...>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
csr2dense(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<HeadId, TailId...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> dense,
        CsrDynamic<HeadId, TailId...> csr)
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

template <class... TensorIndex>
std::ostream& operator<<(std::ostream& os, CsrDynamic<TensorIndex...> const& csr)
{
    os << "----------\n";
    for (std::size_t i = 0; i < csr.coalesc_idx().size(); ++i) {
        os << csr.coalesc_idx()[i] << " ";
    }
    os << "\n";
    for (std::size_t i = 0; i < csr.idx()[0].size(); ++i) {
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
