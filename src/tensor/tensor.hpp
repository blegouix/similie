// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

struct TensorIndex
{
    static const std::size_t s_dim_size;
};

#define SIL_TENSOR_INDEX(NAME, DIM_SIZE)                                                           \
    struct NAME : sil::tensor::TensorIndex                                                         \
    {                                                                                              \
        static const std::size_t s_dim_size;                                                       \
    };                                                                                             \
    std::size_t const NAME::s_dim_size = DIM_SIZE;


template <std::derived_from<TensorIndex>... Index>
class TensorDomain
{
private:
    ddc::DiscreteDomain<Index...> const m_tensor_dom;

public:
    explicit TensorDomain();

    ddc::DiscreteDomain<Index...> operator()();
};

template <std::derived_from<TensorIndex>... Index>
TensorDomain<Index...>::TensorDomain()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::s_dim_size)...))
{
}

template <std::derived_from<TensorIndex>... Index>
ddc::DiscreteDomain<Index...> TensorDomain<Index...>::operator()()
{
    return m_tensor_dom;
}

} // namespace tensor

} // namespace sil
