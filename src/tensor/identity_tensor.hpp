// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include "tensor_impl.hpp"

namespace sil {

namespace tensor {

// struct representing an identity tensor (no storage).
template <class... TensorIndex>
struct IdentityTensorIndex
{
    static constexpr std::size_t rank()
    {
        return (TensorIndex::rank() + ...);
    }

    static constexpr std::size_t size()
    {
        return (TensorIndex::size() * ...);
    }

    static constexpr std::size_t mem_size()
    {
        return 0;
    }

    static constexpr std::size_t access_size()
    {
        return 2;
    }

private:
    template <class Head, class... Tail>
    inline static constexpr bool are_all_same = (std::is_same_v<Head, Tail> && ...);

public:
    template <class... CDim>
    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> mem_id()
    {
        // static_assert(rank() == sizeof...(CDim));
        assert(false);
        return std::pair<
                std::vector<double>,
                std::vector<std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {});
    }

    template <class... CDim>
    static constexpr std::size_t access_id()
    {
        if constexpr (!are_all_same<CDim...>) {
            return 0;
        } else {
            return 1;
        }
    }

    static constexpr std::pair<std::vector<double>, std::vector<std::size_t>> access_id_to_mem_id(
            std::size_t access_id)
    {
        assert(false && "There is no mem_id");
        return std::pair<
                std::vector<double>,
                std::vector<std::size_t>>(std::vector<double> {}, std::vector<std::size_t> {});
    }

    template <class Tensor, class Elem>
    static constexpr Tensor::element_type process_access(
            std::function<typename Tensor::element_type(Tensor, Elem)> access,
            Tensor tensor,
            Elem elem)
    {
        if (elem.uid() == 0) {
            return 0.;
        } else {
            return 1;
        }
    }
};

namespace detail {
template <class... SubIndex>
struct IsTensorIndex<IdentityTensorIndex<SubIndex...>>
{
    using type = std::true_type;
};

} // namespace detail

} // namespace tensor

} // namespace sil
