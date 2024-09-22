// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

template <class... CDim>
struct TensorIndex
{
    static constexpr std::size_t dim_size()
    {
        return sizeof...(CDim);
    }
};

// template <std::derived_from<TensorIndex>... Index>
template <class... Index>
class TensorDomain
{
private:
    ddc::DiscreteDomain<Index...> const m_tensor_dom;

public:
    explicit TensorDomain();

    ddc::DiscreteDomain<Index...> operator()();
};

template <class... Index>
TensorDomain<Index...>::TensorDomain()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::dim_size())...))
{
}

template <class... Index>
ddc::DiscreteDomain<Index...> TensorDomain<Index...>::operator()()
{
    return m_tensor_dom;
}

} // namespace tensor

} // namespace sil
