// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

namespace sil {

namespace tensor {

template <class... CDim>
struct TensorIndex
{
    using type_seq_dimensions = ddc::detail::TypeSeq<CDim...>;

    static constexpr std::size_t dim_size()
    {
        return sizeof...(CDim);
    }
};

// Check that Index derives from TensorIndex despite the templating
template <class Index>
concept TensorIndexConcept = requires(Index index)
{
    []<class... T>(TensorIndex<T...>&) {}(index);
};

template <TensorIndexConcept... Index>
class TensorHandler
{
private:
    ddc::DiscreteDomain<Index...> const m_tensor_dom;

public:
    explicit TensorHandler();

    ddc::DiscreteDomain<Index...> get_domain();

    template <class... CDim>
    ddc::DiscreteElement<Index...> get_element();
};

template <TensorIndexConcept... Index>
TensorHandler<Index...>::TensorHandler()
    : m_tensor_dom(
            ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(0)...),
            ddc::DiscreteVector<Index...>(ddc::DiscreteVector<Index>(Index::dim_size())...))
{
}

template <TensorIndexConcept... Index>
ddc::DiscreteDomain<Index...> TensorHandler<Index...>::get_domain()
{
    return m_tensor_dom;
}

template <TensorIndexConcept... Index>
template <class... CDim>
ddc::DiscreteElement<Index...> TensorHandler<Index...>::get_element()
{
    return ddc::DiscreteElement<Index...>(ddc::DiscreteElement<Index>(
            ddc::type_seq_rank_v<CDim, typename Index::type_seq_dimensions>)...);
}

} // namespace tensor

} // namespace sil
