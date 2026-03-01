// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/ddc.hpp>

#include <similie/misc/are_all_same.hpp>
#include <similie/misc/binomial_coefficient.hpp>
#include <similie/misc/domain_contains.hpp>
#include <similie/misc/filled_struct.hpp>
#include <similie/misc/macros.hpp>
#include <similie/misc/portable_stl.hpp>
#include <similie/misc/specialization.hpp>
#include <similie/misc/type_seq_conversion.hpp>
#include <similie/tensor/antisymmetric_tensor.hpp>
#include <similie/tensor/dummy_index.hpp>
#include <similie/tensor/tensor_impl.hpp>

#include <Kokkos_StdAlgorithms.hpp>

#include "cochain.hpp"
#include "cosimplex.hpp"


namespace sil {

namespace exterior {

namespace detail {

template <class T>
struct CoboundaryType;

template <
        std::size_t K,
        class... Tag,
        class ElementType,
        class LayoutStridedPolicy1,
        class LayoutStridedPolicy2,
        class ExecSpace>
struct CoboundaryType<
        Cochain<Chain<Simplex<K, Tag...>, LayoutStridedPolicy1, ExecSpace>,
                ElementType,
                LayoutStridedPolicy2>>
{
    using type = Cosimplex<Simplex<K + 1, Tag...>, ElementType>;
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
using coboundary_t = typename detail::CoboundaryType<CochainType>::type;

namespace detail {

template <class TagToAddToCochain, class CochainTag>
struct CoboundaryIndex;

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorNatIndex CochainTag>
    requires(CochainTag::rank() == 0)
struct CoboundaryIndex<TagToAddToCochain, CochainTag>
{
    using type = TagToAddToCochain;
};

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorNatIndex CochainTag>
    requires(CochainTag::rank() == 1)
struct CoboundaryIndex<TagToAddToCochain, CochainTag>
{
    using type = tensor::TensorAntisymmetricIndex<TagToAddToCochain, CochainTag>;
};

template <tensor::TensorNatIndex TagToAddToCochain, tensor::TensorNatIndex... Tag>
struct CoboundaryIndex<TagToAddToCochain, tensor::TensorAntisymmetricIndex<Tag...>>
{
    using type = tensor::TensorAntisymmetricIndex<TagToAddToCochain, Tag...>;
};

} // namespace detail

template <class TagToAddToCochain, class CochainTag>
using coboundary_index_t = typename detail::CoboundaryIndex<TagToAddToCochain, CochainTag>::type;

namespace detail {

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
struct CoboundaryTensorType;

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainIndex,
        class ElementType,
        class... DDim,
        class SupportType,
        class MemorySpace>
struct CoboundaryTensorType<
        TagToAddToCochain,
        CochainIndex,
        tensor::Tensor<ElementType, ddc::DiscreteDomain<DDim...>, SupportType, MemorySpace>>
{
    static_assert(ddc::type_seq_contains_v<
                  ddc::detail::TypeSeq<CochainIndex>,
                  ddc::detail::TypeSeq<DDim...>>);
    using type = tensor::Tensor<
            ElementType,
            ddc::replace_dim_of_t<
                    ddc::DiscreteDomain<DDim...>,
                    CochainIndex,
                    coboundary_index_t<TagToAddToCochain, CochainIndex>>,
            SupportType,
            MemorySpace>;
};

} // namespace detail

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
using coboundary_tensor_t =
        typename detail::CoboundaryTensorType<TagToAddToCochain, CochainTag, TensorType>::type;

namespace detail {

template <misc::Specialization<Chain> ChainType>
struct ComputeSimplex;

template <std::size_t K, class... Tag, class LayoutStridedPolicy, class ExecSpace>
struct ComputeSimplex<Chain<Simplex<K, Tag...>, LayoutStridedPolicy, ExecSpace>>
{
    KOKKOS_FUNCTION static Simplex<K + 1, Tag...> run(
            Chain<Simplex<K, Tag...>, LayoutStridedPolicy, ExecSpace> const& chain)
    {
        ddc::DiscreteVector<Tag...> vect {
                0 * ddc::type_seq_rank_v<Tag, ddc::detail::TypeSeq<Tag...>>...};
        for (auto i = chain.begin(); i < chain.end(); ++i) {
            vect = ddc::DiscreteVector<Tag...> {
                    (static_cast<bool>(vect.template get<Tag>())
                     || static_cast<bool>((*i).discrete_vector().template get<Tag>()))...};
        }
        return Simplex(
                std::integral_constant<std::size_t, K + 1> {},
                chain[0].discrete_element(), // This is an assumption on the structure of the chain, which is satisfied if it has been produced using boundary()
                vect);
    }
};

} // namespace detail

template <misc::Specialization<Cochain> CochainType>
KOKKOS_FUNCTION coboundary_t<CochainType> coboundary(
        CochainType
                cochain) // Warning: only cochain.chain() produced using boundary() are supported
{
    assert(cochain.size() == 2 * (cochain.dimension() + 1)
           && "only cochain over the boundary of a single simplex is supported");

    /* Commented because would require an additional buffer
    assert(boundary(cochain.chain()) == boundary_t<typename CochainType::chain_type> {}
           && "only cochain over the boundary of a single simplex is supported");
     */

    return coboundary_t<CochainType>(
            detail::ComputeSimplex<typename CochainType::chain_type>::run(cochain.chain()),
            cochain.integrate());
}

namespace detail {

template <tensor::TensorNatIndex Index, class Dom>
struct NonSpectatorDimension;

template <tensor::TensorNatIndex Index, class... DDim>
struct NonSpectatorDimension<Index, ddc::DiscreteDomain<DDim...>>
{
    using type = ddc::cartesian_prod_t<std::conditional_t<
            ddc::type_seq_contains_v<
                    ddc::detail::TypeSeq<typename DDim::continuous_dimension_type>,
                    typename Index::type_seq_dimensions>,
            ddc::DiscreteDomain<DDim>,
            ddc::DiscreteDomain<>>...>;
};

template <class DDim>
struct CoboundaryRank;

template <class... DDim>
struct CoboundaryRank<ddc::DiscreteDomain<DDim...>>
{
    static constexpr std::size_t value = sizeof...(DDim);
};

template <class DDim>
inline constexpr std::size_t coboundary_rank_v = CoboundaryRank<DDim>::value;

template <class SimplexType>
KOKKOS_FUNCTION void fill_simplex_boundary_half_(
        std::array<boundary_t<SimplexType>, SimplexType::dimension()>& out,
        typename boundary_t<SimplexType>::discrete_element_type elem,
        typename SimplexType::discrete_vector_type vect,
        bool negative = false)
{
    auto array = ddc::detail::array(vect);
    int id_dist = -1;
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        auto array_ = array;
        auto id = array_.begin() + id_dist + 1;
        while (id != array_.end() && *id == 0) {
            ++id;
        }
        assert(id != array_.end());
        id_dist = static_cast<int>(id - array_.begin());
        *id = 0;
        typename SimplexType::discrete_vector_type vect_;
        ddc::detail::array(vect_) = array_;
        out[i] = boundary_t<SimplexType>(elem, vect_, (negative + i) % 2);
    }
}

template <class SimplexType>
KOKKOS_FUNCTION void fill_simplex_boundary_(
        std::array<boundary_t<SimplexType>, 2 * SimplexType::dimension()>& out,
        SimplexType simplex)
{
    std::array<boundary_t<SimplexType>, SimplexType::dimension()> lower_half;
    fill_simplex_boundary_half_<SimplexType>(
            lower_half,
            simplex.discrete_element(),
            simplex.discrete_vector(),
            SimplexType::dimension() % 2);
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        out[i] = lower_half[i];
    }

    std::array<boundary_t<SimplexType>, SimplexType::dimension()> upper_half;
    fill_simplex_boundary_half_<SimplexType>(
            upper_half,
            simplex.discrete_element() + simplex.discrete_vector(),
            -simplex.discrete_vector());
    for (std::size_t i = 0; i < SimplexType::dimension(); ++i) {
        out[i + SimplexType::dimension()] = upper_half[i];
    }

    if constexpr (SimplexType::dimension() % 2 == 0) {
        if (!simplex.negative()) {
            for (std::size_t i = 0; i < 2 * SimplexType::dimension(); ++i) {
                out[i].negative() = !out[i].negative();
            }
        }
    } else {
        if (simplex.negative()) {
            for (std::size_t i = 0; i < 2 * SimplexType::dimension(); ++i) {
                out[i].negative() = !out[i].negative();
            }
        }
    }
}

template <class VectType, std::size_t NDims, std::size_t K, std::size_t NBasis>
KOKKOS_FUNCTION void fill_tangent_basis_(std::array<VectType, NBasis>& basis)
{
    if constexpr (NBasis == 0) {
        return;
    } else if constexpr (K == 0) {
        VectType vect;
        auto array = ddc::detail::array(vect);
        for (std::size_t j = 0; j < NDims; ++j) {
            array[j] = 0;
        }
        basis[0] = vect;
        return;
    } else {
        std::array<int, K> indices {};
        for (std::size_t i = 0; i < K; ++i) {
            indices[i] = static_cast<int>(i);
        }

        std::size_t out_idx = 0;
        while (true) {
            VectType vect;
            auto array = ddc::detail::array(vect);
            for (std::size_t j = 0; j < NDims; ++j) {
                array[j] = 0;
            }
            for (std::size_t j = 0; j < K; ++j) {
                array[static_cast<std::size_t>(indices[j])] = 1;
            }
            basis[out_idx++] = vect;

            int pos = static_cast<int>(K) - 1;
            while (pos >= 0 && indices[static_cast<std::size_t>(pos)] == static_cast<int>(NDims - K + static_cast<std::size_t>(pos))) {
                --pos;
            }
            if (pos < 0) {
                break;
            }
            ++indices[static_cast<std::size_t>(pos)];
            for (std::size_t j = static_cast<std::size_t>(pos) + 1; j < K; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }
}

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType>
KOKKOS_FUNCTION void coboundary(
        typename ddc::remove_dims_of_t<
                typename coboundary_tensor_t<
                        TagToAddToCochain,
                        CochainTag,
                        TensorType>::discrete_domain_type,
                coboundary_index_t<TagToAddToCochain, CochainTag>>::discrete_element_type elem,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    using full_domain_t = ddc::remove_dims_of_t<
            typename coboundary_tensor_t<
                    TagToAddToCochain,
                    CochainTag,
                    TensorType>::discrete_domain_type,
            coboundary_index_t<TagToAddToCochain, CochainTag>>;
    using non_spectator_domain_t = typename detail::NonSpectatorDimension<
            TagToAddToCochain,
            typename TensorType::non_indices_domain_t>::type;
    static constexpr std::size_t n_dims = coboundary_rank_v<non_spectator_domain_t>;
    static constexpr std::size_t k = CochainTag::rank();
    static constexpr std::size_t chain_size = misc::binomial_coefficient(n_dims, k + 1);
    static constexpr std::size_t lower_chain_size = misc::binomial_coefficient(n_dims, k);
    static constexpr std::size_t boundary_size = 2 * (k + 1);

    using simplex_top_t = simplex_for_domain_t<k + 1, full_domain_t>;
    using simplex_boundary_t = boundary_t<simplex_top_t>;
    using non_spectator_simplex_t = simplex_for_domain_t<k + 1, non_spectator_domain_t>;
    using non_spectator_discrete_vector_type = typename non_spectator_simplex_t::discrete_vector_type;

    std::array<non_spectator_discrete_vector_type, chain_size> chain;
    std::array<non_spectator_discrete_vector_type, lower_chain_size> lower_chain;
    fill_tangent_basis_<non_spectator_discrete_vector_type, n_dims, k + 1, chain_size>(chain);
    fill_tangent_basis_<non_spectator_discrete_vector_type, n_dims, k, lower_chain_size>(
            lower_chain);

    std::array<simplex_boundary_t, boundary_size> simplex_boundary;
    std::array<double, boundary_size> boundary_values;

    for (std::size_t i = 0; i < chain_size; ++i) {
        simplex_top_t simplex(
                std::integral_constant<std::size_t, k + 1> {},
                elem,
                chain[i]);

        fill_simplex_boundary_<simplex_top_t>(simplex_boundary, simplex);

        for (std::size_t j = 0; j < boundary_size; ++j) {
            auto const boundary_simplex = simplex_boundary[j];
            auto lower_it = misc::detail::find(
                    lower_chain.begin(),
                    lower_chain.end(),
                    boundary_simplex.discrete_vector());
            assert(lower_it != lower_chain.end());

            boundary_values[j] = tensor.mem(
                    misc::domain_contains(tensor.domain(), boundary_simplex.discrete_element())
                            ? boundary_simplex.discrete_element()
                            : elem,
                    ddc::DiscreteElement<CochainTag>(static_cast<std::size_t>(
                            lower_it - lower_chain.begin())));
        }

        double integral = 0.;
        for (std::size_t j = 0; j < boundary_size; ++j) {
            integral += (simplex_boundary[j].negative() ? -1. : 1.) * boundary_values[j];
        }

        coboundary_tensor.mem(
                elem,
                ddc::DiscreteElement<coboundary_index_t<TagToAddToCochain, CochainTag>>(i))
                = integral;
    }
}

} // namespace detail

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary(
        ExecSpace const& exec_space,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    ddc::DiscreteDomain batch_dom
            = ddc::remove_dims_of<coboundary_index_t<TagToAddToCochain, CochainTag>>(
                    coboundary_tensor.domain());

    // iterate over every node, we will work inside the tangent space associated to each of them
    SIMILIE_DEBUG_LOG("similie_compute_coboundary");
    ddc::parallel_for_each(
            "similie_compute_coboundary",
            exec_space,
            batch_dom,
            KOKKOS_LAMBDA(typename decltype(batch_dom)::discrete_element_type elem) {
                detail::coboundary<TagToAddToCochain, CochainTag, TensorType>(
                        elem,
                        coboundary_tensor,
                        tensor);
            });

    return coboundary_tensor;
}

template <
        tensor::TensorNatIndex TagToAddToCochain,
        tensor::TensorIndex CochainTag,
        misc::Specialization<tensor::Tensor> TensorType,
        class ExecSpace>
coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> deriv(
        ExecSpace const& exec_space,
        coboundary_tensor_t<TagToAddToCochain, CochainTag, TensorType> coboundary_tensor,
        TensorType tensor)
{
    return coboundary<
            TagToAddToCochain,
            CochainTag,
            TensorType>(exec_space, coboundary_tensor, tensor);
}

} // namespace exterior

} // namespace sil
