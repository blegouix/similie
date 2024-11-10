// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <ddc/ddc.hpp>

#include <boost/algorithm/string.hpp>

#include "csr.hpp"
#include "csr_dynamic.hpp"
#include "prime.hpp"
#include "specialization.hpp"
#include "stride.hpp"
#include "tensor_impl.hpp"

namespace sil {

namespace young_tableau {

namespace detail {

template <class T, T... Args>
constexpr T sum(std::integer_sequence<T, Args...> = {})
{
    return (Args + ...);
}

} // namespace detail

template <class... Row>
class YoungTableauSeq
{
public:
    using shape = std::index_sequence<Row::size()...>;

    static constexpr std::size_t rank = detail::sum(shape());
};

namespace detail {

// Extract Row from a YoungTableauSeq at index I
template <std::size_t I, std::size_t Id, class TableauSeq>
struct ExtractRow;

template <std::size_t I, std::size_t Id, class HeadRow, class... TailRow>
struct ExtractRow<I, Id, YoungTableauSeq<HeadRow, TailRow...>>
{
    using type = std::conditional_t<
            I == Id,
            HeadRow,
            typename ExtractRow<I, Id + 1, YoungTableauSeq<TailRow...>>::type>;
};

template <std::size_t I, std::size_t Id>
struct ExtractRow<I, Id, YoungTableauSeq<>>
{
    using type = std::index_sequence<>;
};

template <std::size_t I, class TableauSeq>
using extract_row_t = ExtractRow<I, 0, TableauSeq>::type;

// Override the row of YoungTableauSeq at index I with RowToSet
template <
        std::size_t I,
        std::size_t Id,
        class RowToSet,
        class HeadTableauSeq,
        class InterestTableauSeq,
        class TailTableauSeq>
struct OverrideRow;

template <
        std::size_t I,
        std::size_t Id,
        class RowToSet,
        class... HeadRow,
        class InterestRow,
        class HeadOfTailRow,
        class... TailOfTailRow>
struct OverrideRow<
        I,
        Id,
        RowToSet,
        YoungTableauSeq<HeadRow...>,
        YoungTableauSeq<InterestRow>,
        YoungTableauSeq<HeadOfTailRow, TailOfTailRow...>>
{
    using type = std::conditional_t<
            I == Id,
            YoungTableauSeq<HeadRow..., RowToSet, HeadOfTailRow, TailOfTailRow...>,
            typename OverrideRow<
                    I,
                    Id + 1,
                    RowToSet,
                    YoungTableauSeq<HeadRow..., InterestRow>,
                    YoungTableauSeq<HeadOfTailRow>,
                    YoungTableauSeq<TailOfTailRow...>>::type>;
};

template <std::size_t I, std::size_t Id, class RowToSet, class... HeadRow, class InterestRow>
struct OverrideRow<
        I,
        Id,
        RowToSet,
        YoungTableauSeq<HeadRow...>,
        YoungTableauSeq<InterestRow>,
        YoungTableauSeq<>>
{
    using type = YoungTableauSeq<HeadRow..., RowToSet>;
};

template <std::size_t I, class RowToSet, class TableauSeq>
struct OverrideRowHelper;

template <std::size_t I, class RowToSet, class InterestRow, class... TailRow>
struct OverrideRowHelper<I, RowToSet, YoungTableauSeq<InterestRow, TailRow...>>
{
    using type = OverrideRow<
            I,
            0,
            RowToSet,
            YoungTableauSeq<>,
            YoungTableauSeq<InterestRow>,
            YoungTableauSeq<TailRow...>>::type;
};

template <std::size_t I, class RowToSet, class TableauSeq>
using override_row_t = OverrideRowHelper<I, RowToSet, TableauSeq>::type;

// Add cell with value Value at the end of a row
template <class Row, std::size_t Value>
struct AddCellToRow;

template <std::size_t... Elem, std::size_t Value>
struct AddCellToRow<std::index_sequence<Elem...>, Value>
{
    using type = std::index_sequence<Elem..., Value>;
};

template <class Row, std::size_t Value>
using add_cell_to_row_t = AddCellToRow<Row, Value>::type;

// Add cell with value Value at the end of the row at index I
template <class Tableau, std::size_t I, std::size_t Value>
struct AddCellToTableau;

template <class... Row, std::size_t I, std::size_t Value>
struct AddCellToTableau<YoungTableauSeq<Row...>, I, Value>
{
    using type = detail::override_row_t<
            I,
            add_cell_to_row_t<extract_row_t<I, YoungTableauSeq<Row...>>, Value>,
            YoungTableauSeq<Row...>>;
};

template <class Tableau, std::size_t I, std::size_t Value>
using add_cell_to_tableau_t = AddCellToTableau<Tableau, I, Value>::type;

// Compute dual of YoungTableauSeq
template <class TableauDual, class Tableau, std::size_t I, std::size_t J>
struct Dual;

template <
        class TableauDual,
        std::size_t HeadElemOfHeadRow,
        std::size_t... TailElemOfHeadRow,
        class... TailRow,
        std::size_t I,
        std::size_t J>
struct Dual<
        TableauDual,
        YoungTableauSeq<std::index_sequence<HeadElemOfHeadRow, TailElemOfHeadRow...>, TailRow...>,
        I,
        J>
{
    using type = std::conditional_t<
            sizeof...(TailRow) == 0 && sizeof...(TailElemOfHeadRow) == 0,
            add_cell_to_tableau_t<TableauDual, J, HeadElemOfHeadRow>,
            std::conditional_t<
                    sizeof...(TailElemOfHeadRow) == 0,
                    typename Dual<
                            add_cell_to_tableau_t<TableauDual, J, HeadElemOfHeadRow>,
                            YoungTableauSeq<TailRow...>,
                            I + 1,
                            0>::type,
                    typename Dual<
                            add_cell_to_tableau_t<TableauDual, J, HeadElemOfHeadRow>,
                            YoungTableauSeq<std::index_sequence<TailElemOfHeadRow...>, TailRow...>,
                            I,
                            J + 1>::type>>;
};

template <class TableauDual, class... TailRow, std::size_t I, std::size_t J>
struct Dual<TableauDual, YoungTableauSeq<std::index_sequence<>, TailRow...>, I, J>
{
    using type = YoungTableauSeq<>;
};

template <class TableauDual, std::size_t I, std::size_t J>
struct Dual<TableauDual, YoungTableauSeq<>, I, J>
{
    using type = TableauDual;
};

template <class Tableau>
struct DualHelper;

template <std::size_t... ElemOfHeadRow, class... TailRow>
struct DualHelper<YoungTableauSeq<std::index_sequence<ElemOfHeadRow...>, TailRow...>>
{
    using type
            = Dual<YoungTableauSeq<std::conditional_t<
                           ElemOfHeadRow == -1,
                           std::index_sequence<>,
                           std::index_sequence<>>...>,
                   YoungTableauSeq<std::index_sequence<ElemOfHeadRow...>, TailRow...>,
                   0,
                   0>::type;
};

template <>
struct DualHelper<YoungTableauSeq<>>
{
    using type = YoungTableauSeq<>;
};

template <class Tableau>
using dual_t = DualHelper<Tableau>::type;

// Compute hooks
template <class TableauHooks, class Tableau, class Shape, std::size_t I, std::size_t J>
struct Hooks;

template <
        class TableauHooks,
        std::size_t HeadElemOfHeadRow,
        std::size_t... TailElemOfHeadRow,
        class... TailRow,
        std::size_t HeadRowSize,
        std::size_t... TailRowSize,
        std::size_t I,
        std::size_t J>
struct Hooks<
        TableauHooks,
        YoungTableauSeq<std::index_sequence<HeadElemOfHeadRow, TailElemOfHeadRow...>, TailRow...>,
        std::index_sequence<HeadRowSize, TailRowSize...>,
        I,
        J>
{
    using type = std::conditional_t<
            sizeof...(TailRow) == 0 && sizeof...(TailElemOfHeadRow) == 0,
            add_cell_to_tableau_t<TableauHooks, I, HeadRowSize - J>,
            std::conditional_t<
                    sizeof...(TailElemOfHeadRow) == 0,
                    typename Hooks<
                            add_cell_to_tableau_t<TableauHooks, I, HeadRowSize - J>,
                            YoungTableauSeq<TailRow...>,
                            std::index_sequence<TailRowSize...>,
                            I + 1,
                            0>::type,
                    typename Hooks<
                            add_cell_to_tableau_t<TableauHooks, I, HeadRowSize - J>,
                            YoungTableauSeq<std::index_sequence<TailElemOfHeadRow...>, TailRow...>,
                            std::index_sequence<HeadRowSize, TailRowSize...>,
                            I,
                            J + 1>::type>>;
};

template <class TableauHooks, class... TailRow, class Shape, std::size_t I, std::size_t J>
struct Hooks<TableauHooks, YoungTableauSeq<std::index_sequence<>, TailRow...>, Shape, I, J>
{
    using type = YoungTableauSeq<>;
};

template <class TableauHooks, class Shape, std::size_t I, std::size_t J>
struct Hooks<TableauHooks, YoungTableauSeq<>, Shape, I, J>
{
    using type = TableauHooks;
};

template <class Tableau>
struct HooksHelper;

template <class... Row>
struct HooksHelper<YoungTableauSeq<Row...>>
{
    using type
            = Hooks<YoungTableauSeq<std::conditional_t<true, std::index_sequence<>, Row>...>,
                    YoungTableauSeq<Row...>,
                    typename YoungTableauSeq<Row...>::shape,
                    0,
                    0>::type;
};

template <>
struct HooksHelper<YoungTableauSeq<>>
{
    using type = YoungTableauSeq<>;
};

template <class Tableau>
using hooks_t = HooksHelper<Tableau>::type;

// Sum the partial contributions of hooks to get hook lengths (= hooks+hooks_of_dual^T-1)
template <class TableauHookLengths, class Tableau1, class Tableau2, std::size_t I>
struct HookLengths;

template <
        class TableauHookLengths,
        std::size_t HeadElemOfHeadRow1,
        std::size_t... TailElemOfHeadRow1,
        class... TailRow1,
        std::size_t HeadElemOfHeadRow2,
        std::size_t... TailElemOfHeadRow2,
        class... TailRow2,
        std::size_t I>
struct HookLengths<
        TableauHookLengths,
        YoungTableauSeq<
                std::index_sequence<HeadElemOfHeadRow1, TailElemOfHeadRow1...>,
                TailRow1...>,
        YoungTableauSeq<
                std::index_sequence<HeadElemOfHeadRow2, TailElemOfHeadRow2...>,
                TailRow2...>,
        I>
{
    using type = std::conditional_t<
            sizeof...(TailRow1) == 0 && sizeof...(TailElemOfHeadRow1) == 0,
            add_cell_to_tableau_t<
                    TableauHookLengths,
                    I,
                    HeadElemOfHeadRow1 + HeadElemOfHeadRow2 - 1>,
            std::conditional_t<
                    sizeof...(TailElemOfHeadRow1) == 0,
                    typename HookLengths<
                            add_cell_to_tableau_t<
                                    TableauHookLengths,
                                    I,
                                    HeadElemOfHeadRow1 + HeadElemOfHeadRow2 - 1>,
                            YoungTableauSeq<TailRow1...>,
                            YoungTableauSeq<TailRow2...>,
                            I + 1>::type,
                    typename HookLengths<
                            add_cell_to_tableau_t<
                                    TableauHookLengths,
                                    I,
                                    HeadElemOfHeadRow1 + HeadElemOfHeadRow2 - 1>,
                            YoungTableauSeq<
                                    std::index_sequence<TailElemOfHeadRow1...>,
                                    TailRow1...>,
                            YoungTableauSeq<
                                    std::index_sequence<TailElemOfHeadRow2...>,
                                    TailRow2...>,
                            I>::type>>;
};

template <class TableauHookLengths, class... TailRow1, class... TailRow2, std::size_t I>
struct HookLengths<
        TableauHookLengths,
        YoungTableauSeq<std::index_sequence<>, TailRow1...>,
        YoungTableauSeq<std::index_sequence<>, TailRow2...>,
        I>
{
    using type = YoungTableauSeq<>;
};

template <class TableauHookLengths, std::size_t I>
struct HookLengths<TableauHookLengths, YoungTableauSeq<>, YoungTableauSeq<>, I>
{
    using type = TableauHookLengths;
};

template <class Tableau1, class Tableau2>
struct HookLengthsHelper;

template <class... Row1, class Tableau2>
struct HookLengthsHelper<YoungTableauSeq<Row1...>, Tableau2>
{
    using type = HookLengths<
            YoungTableauSeq<std::conditional_t<true, std::index_sequence<>, Row1>...>,
            YoungTableauSeq<Row1...>,
            Tableau2,
            0>::type;
};

template <>
struct HookLengthsHelper<YoungTableauSeq<>, YoungTableauSeq<>>
{
    using type = YoungTableauSeq<>;
};

template <class Tableau1, class Tableau2>
using hook_lengths_t = HookLengthsHelper<Tableau1, Tableau2>::type;

// Compute irreductible representations dimension
template <std::size_t Dimension, class TableauHookLengths, std::size_t I, std::size_t J>
struct IrrepDim;

template <
        std::size_t Dimension,
        std::size_t HeadElemOfHeadRow,
        std::size_t... TailElemOfHeadRow,
        class... TailRow,
        std::size_t I,
        std::size_t J>
struct IrrepDim<
        Dimension,
        YoungTableauSeq<std::index_sequence<HeadElemOfHeadRow, TailElemOfHeadRow...>, TailRow...>,
        I,
        J>
{
    static consteval std::size_t run(double prod)
    {
        prod *= Dimension + J - I;
        prod /= HeadElemOfHeadRow;
        if constexpr (sizeof...(TailRow) == 0 && sizeof...(TailElemOfHeadRow) == 0) {
            return prod;
        } else if (sizeof...(TailElemOfHeadRow) == 0) {
            return IrrepDim<Dimension, YoungTableauSeq<TailRow...>, I + 1, 0>::run(prod);
        } else {
            return IrrepDim<
                    Dimension,
                    YoungTableauSeq<std::index_sequence<TailElemOfHeadRow...>, TailRow...>,
                    I,
                    J + 1>::run(prod);
        }
    }
};

} // namespace detail

/**
 * YoungTableau class
 */
template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
class YoungTableau
{
public:
    using tableau_seq = TableauSeq;
    using shape = typename TableauSeq::shape;

private:
    static constexpr std::size_t s_d = Dimension;
    static constexpr std::size_t s_r = TableauSeq::rank;

public:
    using dual = YoungTableau<s_d, detail::dual_t<tableau_seq>>;
    using hook_lengths = detail::hook_lengths_t<
            detail::hooks_t<tableau_seq>,
            detail::dual_t<detail::hooks_t<detail::dual_t<tableau_seq>>>>;

private:
    static constexpr std::size_t s_irrep_dim = detail::IrrepDim<s_d, hook_lengths, 0, 0>::run(1);

    static constexpr std::string generate_tag();

    static constexpr std::string s_tag = generate_tag();

    static consteval auto load_irrep();

    static constexpr auto s_irrep = load_irrep();

public:
    YoungTableau();

    static constexpr std::size_t dimension()
    {
        return s_d;
    }

    static constexpr std::size_t rank()
    {
        return s_r;
    }

    static constexpr std::size_t irrep_dim()
    {
        return s_irrep_dim;
    }

    static constexpr std::string tag()
    {
        return s_tag;
    }

private:
    static constexpr std::size_t n_nonzeros_in_irrep()
    {
        return std::get<2>(std::get<0>(s_irrep)).size();
    }

public:
    template <tensor::TensorNatIndex... Id>
    using projector_domain = ddc::DiscreteDomain<tensor::prime<Id>..., Id...>;

    template <tensor::TensorNatIndex... Id>
    static auto projector();

    template <tensor::TensorIndex BasisId, tensor::TensorNatIndex... Id>
    static constexpr csr::Csr<n_nonzeros_in_irrep(), BasisId, Id...> u(
            ddc::DiscreteDomain<Id...> restricted_domain);

    template <tensor::TensorIndex BasisId, tensor::TensorNatIndex... Id>
    static constexpr csr::Csr<n_nonzeros_in_irrep(), BasisId, Id...> v(
            ddc::DiscreteDomain<Id...> restricted_domain);
};

namespace detail {

template <std::size_t Dimension, class... TailRow, std::size_t I, std::size_t J>
struct IrrepDim<Dimension, YoungTableauSeq<std::index_sequence<>, TailRow...>, I, J>
{
    static consteval std::size_t run(double prod)
    {
        return prod;
    }
};

template <std::size_t Dimension, std::size_t I, std::size_t J>
struct IrrepDim<Dimension, YoungTableauSeq<>, I, J>
{
    static consteval std::size_t run(double prod)
    {
        return prod;
    }
};

// Type of index used by tensor::OrthonormalBasisSubspaceEigenvalueOne
template <std::size_t I>
struct Dummy
{
};

template <class Ids>
struct NaturalIndex;

template <std::size_t... Id>
struct NaturalIndex<std::index_sequence<Id...>>
{
    template <std::size_t RankId>
    struct type : tensor::TensorNaturalIndex<Dummy<Id>...>
    {
    };
};

template <class NaturalIds, class RankIds>
struct DummyIndex;

template <std::size_t... Id, std::size_t... RankId>
struct DummyIndex<std::index_sequence<Id...>, std::index_sequence<RankId...>>
{
    using type = tensor::TensorFullIndex<
            typename NaturalIndex<std::index_sequence<Id...>>::template type<RankId>...>;
};

template <std::size_t Dimension, std::size_t Rank>
using dummy_index_t
        = DummyIndex<std::make_index_sequence<Dimension>, std::make_index_sequence<Rank>>::type;

// Gram-Schmidt approach to produce from vec an orthogonal vector to the vector space generated by basis
template <
        class ElementType,
        class... Id,
        class LayoutStridedPolicy,
        class MemorySpace,
        class BasisId>
tensor::Tensor<ElementType, ddc::DiscreteDomain<Id...>, LayoutStridedPolicy, MemorySpace>
orthogonalize(
        tensor::Tensor<ElementType, ddc::DiscreteDomain<Id...>, LayoutStridedPolicy, MemorySpace>
                tensor,
        csr::CsrDynamic<BasisId, Id...> basis,
        std::size_t max_basis_id)
{
    ddc::Chunk eigentensor_alloc(
            ddc::DiscreteDomain<BasisId, Id...>(basis.domain()),
            ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::DiscreteDomain<BasisId, Id...>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            eigentensor(eigentensor_alloc);
    for (ddc::DiscreteElement<BasisId> elem : ddc::DiscreteDomain<BasisId>(
                 ddc::DiscreteElement<BasisId>(0),
                 ddc::DiscreteVector<BasisId>(max_basis_id))) {
        csr::csr2dense(eigentensor, basis.get(elem));

        ddc::Chunk scalar_prod_alloc(ddc::DiscreteDomain<> {}, ddc::HostAllocator<double>());
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                scalar_prod(scalar_prod_alloc);
        tensor::tensor_prod(scalar_prod, tensor, eigentensor[ddc::DiscreteElement<BasisId>(0)]);
        ddc::Chunk norm_squared_alloc(ddc::DiscreteDomain<> {}, ddc::HostAllocator<double>());
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                norm_squared(norm_squared_alloc);
        tensor::tensor_prod(norm_squared, eigentensor, eigentensor);

        eigentensor *= -1. * scalar_prod(ddc::DiscreteElement<>())
                       / norm_squared(ddc::DiscreteElement<>());
        tensor += eigentensor[ddc::DiscreteElement<BasisId>(0)];
    }
    return tensor;
}

// Function to find the number at the given index `n` with increasing Hamming weight
std::vector<bool> index_hamming_weight_code(std::size_t index, std::size_t length)
{
    std::size_t count = 0;
    std::vector<bool> bits(length);
    for (std::size_t hamming_weight = 0; hamming_weight < length; ++hamming_weight) {
        std::fill(bits.begin(), bits.begin() + hamming_weight, true);
        std::fill(bits.begin() + hamming_weight, bits.end(), false);
        do {
            if (count == index) {
                return bits;
            }
            count++;
        } while (std::prev_permutation(bits.begin(), bits.end()));
    }
    assert(false && "hamming weight code not found for index");
    return bits;
}

// Dummy tag used by OrthonormalBasisSubspaceEigenvalueOne (coalescent dimension of the CsrDynamic storage)
struct BasisId : tensor::TensorNaturalIndex<>
{
};

template <class Ids>
struct OrthonormalBasisSubspaceEigenvalueOne;

template <class... Id>
struct OrthonormalBasisSubspaceEigenvalueOne<tensor::TensorFullIndex<Id...>>
{
    template <class YoungTableau>
    static std::pair<csr::CsrDynamic<BasisId, Id...>, csr::CsrDynamic<BasisId, Id...>> run(
            YoungTableau tableau)
    {
        auto [proj_alloc, proj] = tableau.template projector<Id...>();

        tensor::TensorAccessor<Id...> candidate_accessor;
        ddc::DiscreteDomain<Id...> candidate_dom = candidate_accessor.mem_domain();
        ddc::Chunk candidate_alloc(candidate_dom, ddc::HostAllocator<double>());
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                candidate(candidate_alloc);
        ddc::Chunk prod_alloc(
                tensor::natural_tensor_prod_domain(proj.domain(), candidate.domain()),
                ddc::HostAllocator<double>());
        tensor::Tensor<
                double,
                tensor::natural_tensor_prod_domain_t<
                        typename YoungTableau::template projector_domain<Id...>,
                        ddc::DiscreteDomain<Id...>>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                prod(prod_alloc);

        ddc::DiscreteDomain<BasisId, Id...> basis_dom(
                ddc::DiscreteDomain<BasisId>(
                        ddc::DiscreteElement<BasisId>(0),
                        ddc::DiscreteVector<BasisId>(tableau.irrep_dim())),
                candidate_dom);
        csr::CsrDynamic<BasisId, Id...> u(basis_dom);
        csr::CsrDynamic<BasisId, Id...> v(basis_dom);
        std::size_t n_irreps = 0;
        std::size_t index = 0;
        while (n_irreps < tableau.irrep_dim()) {
            index++;
            std::vector<bool> hamming_weight_code
                    = index_hamming_weight_code(index, (Id::size() * ...));

            ddc::parallel_for_each(candidate.domain(), [&](ddc::DiscreteElement<Id...> elem) {
                candidate(elem) = hamming_weight_code[(
                        (misc::detail::stride<Id, Id...>() * elem.template uid<Id>()) + ...)];
            });

            tensor::tensor_prod(prod, proj, candidate);
            Kokkos::deep_copy(
                    candidate.allocation_kokkos_view(),
                    prod.allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions

            orthogonalize(candidate, v, n_irreps);

            if (ddc ::transform_reduce(
                        candidate.domain(),
                        false,
                        ddc::reducer::lor<bool>(),
                        [&](ddc::DiscreteElement<Id...> elem) { return candidate(elem) > 1e-6; })) {
                ddc::Chunk
                        norm_squared_alloc(ddc::DiscreteDomain<> {}, ddc::HostAllocator<double>());
                tensor::Tensor<
                        double,
                        ddc::DiscreteDomain<>,
                        Kokkos::layout_right,
                        Kokkos::DefaultHostExecutionSpace::memory_space>
                        norm_squared(norm_squared_alloc);
                tensor::tensor_prod(norm_squared, candidate, candidate);
                ddc::parallel_for_each(candidate.domain(), [&](ddc::DiscreteElement<Id...> elem) {
                    candidate(elem) /= Kokkos::sqrt(norm_squared(ddc::DiscreteElement<>()));
                });
                // Not sure if u = v is correct in any case (ie. complex tensors ?)
                u.push_back(candidate);
                v.push_back(candidate);
                n_irreps++;
                std::cout << n_irreps << "/" << tableau.irrep_dim()
                          << " eigentensors found associated to the eigenvalue 1 for the Young "
                             "projector labelized "
                          << tableau.tag() << std::endl;
            }
        }
        return std::pair<csr::CsrDynamic<BasisId, Id...>, csr::CsrDynamic<BasisId, Id...>>(u, v);
    }
};

} // namespace detail

template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
YoungTableau<Dimension, TableauSeq>::YoungTableau()
{
    // Check if the irrep is available in the dictionnary
    {
        std::ifstream file(IRREPS_DICT_PATH, std::ios::out | std::ios::binary);
        std::string line;
        while (!file.eof()) {
            getline(file, line);
            if (line == s_tag) {
                file.close();
                if (n_nonzeros_in_irrep() == 0) {
                    std::cout << "\033[1;31mIrrep " << s_tag << " in dimension " << s_d
                              << " required and found in dictionnary " << IRREPS_DICT_PATH
                              << " but the executable has been compiled without it. Please "
                                 "recompile.\033[0m"
                              << std::endl;
                }
                return;
            }
        }
    }

    // If the current irrep is not found in the dictionnary, compute and dump it
    std::cout << "\033[1;31mIrrep " << s_tag << " corresponding to the Young Tableau:\033[0m\n"
              << *this << "\n\033[1;31min dimension " << s_d
              << " required but not found in dictionnary " << IRREPS_DICT_PATH
              << ". It will be computed, and you will have to recompile once it is done.\033[0m"
              << std::endl;

    auto [u, v]
            = detail::OrthonormalBasisSubspaceEigenvalueOne<detail::dummy_index_t<s_d, s_r>>::run(
                    *this);

    std::ofstream file(IRREPS_DICT_PATH, std::ios::app | std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << IRREPS_DICT_PATH << std::endl;
        return;
    }
    file << s_tag << "\n";
    u.write(file);
    v.write(file);
    file << "\n";
    file.close();
    if (!file.good()) {
        std::cerr << "Error occurred while writing to file " << IRREPS_DICT_PATH
                  << " while adding irrep " << s_tag << std::endl;
    } else {
        std::cout << "\033[1;32mIrrep " << s_tag << " added to the dictionnary " << IRREPS_DICT_PATH
                  << ".\033[0m \033[1;31mPlease recompile.\033[0m" << std::endl;
    }
}

namespace detail {

// Build index for symmetrizer (such that sym*proj is properly defined)
template <class OId, class... Id>
using symmetrizer_index_t = std::conditional_t<
        (ddc::type_seq_rank_v<OId, ddc::detail::TypeSeq<Id...>> < (sizeof...(Id) / 2)),
        tensor::prime<OId>,
        ddc::type_seq_element_t<
                static_cast<std::size_t>(
                        std::
                                max(static_cast<std::ptrdiff_t>(0),
                                    static_cast<std::ptrdiff_t>(
                                            ddc::type_seq_rank_v<OId, ddc::detail::TypeSeq<Id...>>
                                            - (sizeof...(Id) / 2)))),
                ddc::detail::TypeSeq<Id...>>>;

// Functor to fill identity or transpose projectors
template <std::size_t Dimension, std::size_t Rank, class... NaturalId>
class TrFunctor
{
private:
    using TensorType = tensor::Tensor<
            double,
            ddc::DiscreteDomain<NaturalId...>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>;

    TensorType t;
    std::array<std::size_t, Rank> idx_to_permute;

public:
    TrFunctor(TensorType t_, std::array<std::size_t, Rank> idx_to_permute_)
        : t(t_)
        , idx_to_permute(idx_to_permute_)
    {
    }

    void operator()(ddc::DiscreteElement<NaturalId...> elem) const
    {
        std::array<std::size_t, sizeof...(NaturalId)> elem_array = ddc::detail::array(elem);
        bool has_to_be_one = true;
        for (std::size_t i = 0; i < Rank; ++i) {
            has_to_be_one
                    = has_to_be_one && (elem_array[i] == elem_array[idx_to_permute[i] + Rank]);
        }
        if (has_to_be_one) {
            t(elem) = 1;
        }
    }
};


/*
 Given a permutation of the digits 0..N, 
 returns its parity (or sign): +1 for even parity; -1 for odd.
 */
template <std::size_t Nt>
static int permutation_parity(std::array<std::size_t, Nt> lst)
{
    int parity = 1;
    for (int i = 0; i < lst.size() - 1; ++i) {
        if (lst[i] != i) {
            parity *= -1;
            std::size_t mn
                    = std::distance(lst.begin(), std::min_element(lst.begin() + i, lst.end()));
            std::swap(lst[i], lst[mn]);
        }
    }
    return parity;
}

template <std::size_t Dimension, bool AntiSym, class... Id>
static tensor::Tensor<
        double,
        ddc::DiscreteDomain<Id...>,
        Kokkos::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
fill_symmetrizer(
        tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                Kokkos::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> sym,
        std::array<std::size_t, sizeof...(Id) / 2> idx_to_permute)
{
    ddc::Chunk tr_alloc(sym.domain(), ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::DiscreteDomain<Id...>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tr(tr_alloc);
    ddc::parallel_fill(tr, 0);
    ddc::for_each(tr.domain(), TrFunctor<Dimension, sizeof...(Id) / 2, Id...>(tr, idx_to_permute));

    if constexpr (AntiSym) {
        tr *= static_cast<double>(permutation_parity(idx_to_permute));
    }
    sym += tr;

    return sym;
}

/*
 Compute all permutations on a subset of indices in idx_to_permute, keep the
 rest of the indices where they are.
 */
template <std::size_t Nt, std::size_t Ns>
static std::vector<std::array<std::size_t, Nt>> permutations_subset(
        std::array<std::size_t, Nt> t,
        std::array<std::size_t, Ns> subset_values)
{
    std::array<std::size_t, Ns> subset_indices;
    std::array<std::size_t, Ns> elements_to_permute;
    int j = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (std::find(subset_values.begin(), subset_values.end(), t[i] + 1)
            != std::end(subset_values)) {
            subset_indices[j] = i;
            elements_to_permute[j++] = t[i];
        }
    }

    std::vector<std::array<std::size_t, Nt>> result;
    do {
        std::array<std::size_t, Nt> tmp = t;
        for (std::size_t i = 0; i < Ns; ++i) {
            tmp[subset_indices[i]] = elements_to_permute[i];
        }
        result.push_back(tmp);
    } while (std::next_permutation(elements_to_permute.begin(), elements_to_permute.end()));

    return result;
}

// Compute projector
template <class PartialTableauSeq, std::size_t Dimension, bool AntiSym = false>
struct Projector;

template <std::size_t... ElemOfHeadRow, class... TailRow, std::size_t Dimension, bool AntiSym>
struct Projector<
        YoungTableauSeq<std::index_sequence<ElemOfHeadRow...>, TailRow...>,
        Dimension,
        AntiSym>
{
    template <class... Id>
    static tensor::Tensor<
            double,
            ddc::DiscreteDomain<Id...>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
    run(tensor::Tensor<
            double,
            ddc::DiscreteDomain<Id...>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space> proj)
    {
        if constexpr (sizeof...(ElemOfHeadRow) >= 2) {
            // Allocate & build a symmetric projector for the row
            tensor::TensorAccessor<symmetrizer_index_t<Id, Id...>...> sym_accessor;
            ddc::DiscreteDomain<symmetrizer_index_t<Id, Id...>...> sym_dom
                    = sym_accessor.mem_domain();

            ddc::Chunk sym_alloc(sym_dom, ddc::HostAllocator<double>());
            tensor::Tensor<
                    double,
                    ddc::DiscreteDomain<symmetrizer_index_t<Id, Id...>...>,
                    Kokkos::layout_right,
                    Kokkos::DefaultHostExecutionSpace::memory_space>
                    sym(sym_alloc);
            ddc::parallel_fill(sym, 0);
            std::array<std::size_t, sizeof...(Id) / 2> idx_to_permute;
            for (std::size_t i = 0; i < sizeof...(Id) / 2; ++i) {
                idx_to_permute[i] = i;
            }
            std::array<std::size_t, sizeof...(ElemOfHeadRow)> row_values {ElemOfHeadRow...};
            auto idx_permutations = detail::permutations_subset(
                    idx_to_permute,
                    row_values); // TODO check https://indico.cern.ch/event/814040/contributions/3452485/attachments/1860434/3057354/psr_alcock.pdf page 6, it may be incomplete
            for (std::size_t i = 0; i < idx_permutations.size(); ++i) {
                fill_symmetrizer<
                        Dimension,
                        AntiSym,
                        symmetrizer_index_t<Id, Id...>...>(sym, idx_permutations[i]);
            }

            // Extract the symmetric part (for the row) of the projector (requires an intermediate prod tensor)
            ddc::Chunk prod_alloc(
                    natural_tensor_prod_domain(sym.domain(), proj.domain()),
                    ddc::HostAllocator<double>());
            tensor::Tensor<
                    double,
                    tensor::natural_tensor_prod_domain_t<
                            ddc::DiscreteDomain<symmetrizer_index_t<Id, Id...>...>,
                            ddc::DiscreteDomain<Id...>>,
                    Kokkos::layout_right,
                    Kokkos::DefaultHostExecutionSpace::memory_space>
                    prod(prod_alloc);
            tensor::tensor_prod(prod, sym, proj);
            Kokkos::deep_copy(
                    proj.allocation_kokkos_view(),
                    prod.allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions
        }
        if constexpr (sizeof...(TailRow) == 0) {
            return proj;
        } else {
            return Projector<YoungTableauSeq<TailRow...>, Dimension, AntiSym>::run(proj);
        }
    }
};

} // namespace detail

template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
template <tensor::TensorNatIndex... Id>
auto YoungTableau<Dimension, TableauSeq>::projector()
{
    static_assert(sizeof...(Id) == s_r);
    tensor::TensorAccessor<tensor::prime<Id>..., Id...> proj_accessor;
    ddc::DiscreteDomain<tensor::prime<Id>..., Id...> proj_dom = proj_accessor.mem_domain();

    // Allocate a projector and fill it as an identity tensor
    ddc::Chunk proj_alloc(proj_dom, ddc::HostAllocator<double>());
    tensor::Tensor<
            double,
            ddc::DiscreteDomain<tensor::prime<Id>..., Id...>,
            Kokkos::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            proj(proj_alloc);
    ddc::parallel_fill(proj, 0);
    std::array<std::size_t, s_r> idx_to_permute;
    for (std::size_t i = 0; i < s_r; ++i) {
        idx_to_permute[i] = i;
    }
    ddc::for_each(
            proj.domain(),
            detail::TrFunctor<s_d, s_r, tensor::prime<Id>..., Id...>(proj, idx_to_permute));

    // Build the projector
    detail::Projector<tableau_seq, s_d>::run(proj);
    detail::Projector<typename dual::tableau_seq, s_d, true>::run(proj);
    return std::make_tuple(std::move(proj_alloc), proj);
}

// Access to u and v Csr, allowing to ie. compress or uncompress a tensor with internal symmetries
template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
template <tensor::TensorIndex BasisId, tensor::TensorNatIndex... Id>
constexpr csr::Csr<YoungTableau<Dimension, TableauSeq>::n_nonzeros_in_irrep(), BasisId, Id...>
YoungTableau<Dimension, TableauSeq>::u(ddc::DiscreteDomain<Id...> restricted_domain)
{
    if constexpr (n_nonzeros_in_irrep() != 0) {
        ddc::DiscreteDomain<BasisId, Id...>
                domain(ddc::DiscreteDomain<BasisId>(
                               ddc::DiscreteElement<BasisId>(0),
                               ddc::DiscreteVector<BasisId>(n_nonzeros_in_irrep())),
                       restricted_domain);

        return csr::Csr<n_nonzeros_in_irrep(), BasisId, Id...>(
                domain,
                std::get<0>(std::get<0>(s_irrep)),
                std::get<1>(std::get<0>(s_irrep)),
                std::get<2>(std::get<0>(s_irrep)));
    } else {
        ddc::DiscreteDomain<BasisId, Id...>
                domain(ddc::DiscreteDomain<BasisId>(
                               ddc::DiscreteElement<BasisId>(0),
                               ddc::DiscreteVector<BasisId>(0)),
                       restricted_domain);

        return csr::Csr<n_nonzeros_in_irrep(), BasisId, Id...>(
                domain,
                std::array<std::size_t, BasisId::mem_size() + 1> {},
                std::get<1>(std::get<0>(s_irrep)),
                std::get<2>(std::get<0>(s_irrep)));
    }
}

template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
template <tensor::TensorIndex BasisId, tensor::TensorNatIndex... Id>
constexpr csr::Csr<YoungTableau<Dimension, TableauSeq>::n_nonzeros_in_irrep(), BasisId, Id...>
YoungTableau<Dimension, TableauSeq>::v(ddc::DiscreteDomain<Id...> restricted_domain)
{
    if constexpr (n_nonzeros_in_irrep() != 0) {
        ddc::DiscreteDomain<BasisId, Id...>
                domain(ddc::DiscreteDomain<BasisId>(
                               ddc::DiscreteElement<BasisId>(0),
                               ddc::DiscreteVector<BasisId>(n_nonzeros_in_irrep())),
                       restricted_domain);

        return csr::Csr<n_nonzeros_in_irrep(), BasisId, Id...>(
                domain,
                std::get<0>(std::get<1>(s_irrep)),
                std::get<1>(std::get<1>(s_irrep)),
                std::get<2>(std::get<1>(s_irrep)));
    } else {
        ddc::DiscreteDomain<BasisId, Id...>
                domain(ddc::DiscreteDomain<BasisId>(
                               ddc::DiscreteElement<BasisId>(0),
                               ddc::DiscreteVector<BasisId>(0)),
                       restricted_domain);

        return csr::Csr<n_nonzeros_in_irrep(), BasisId, Id...>(
                domain,
                std::array<std::size_t, BasisId::mem_size() + 1> {},
                std::get<1>(std::get<1>(s_irrep)),
                std::get<2>(std::get<1>(s_irrep)));
    }
}

// Load binary files and build u and v static constexpr Csr at compile-time
namespace detail {

template <std::size_t Line>
consteval std::string_view load_irrep_line_for_tag(std::string_view const tag)
{
    constexpr static char raw[] = {
#embed IRREPS_DICT_PATH
    };

    constexpr static std::string_view str(raw, sizeof(raw));

    size_t tagPos = str.find(tag);

    if (tagPos != std::string::npos) {
        std::size_t endOfTagLine = str.find('\n', tagPos);
        if (endOfTagLine != std::string::npos) {
            std::size_t nextLineStart = endOfTagLine + 1;
            for (std::size_t i = 0; i < Line; ++i) {
                nextLineStart = str.find("break\n", nextLineStart) + 6;
            }
            std::size_t nextLineEnd = str.find("break\n", nextLineStart);

            return std::string_view(str.data() + nextLineStart, nextLineEnd - nextLineStart);
        }
    }

    return "";
}

template <class Ids, std::size_t Offset>
struct LoadIrrepIdxForTag;

template <std::size_t... I, std::size_t Offset>
struct LoadIrrepIdxForTag<std::index_sequence<I...>, Offset>
{
    static consteval std::array<std::string_view, sizeof...(I)> run(std::string_view const tag)
    {
        return std::array<std::string_view, sizeof...(I)> {
                load_irrep_line_for_tag<I + Offset>(tag)...};
    }
};

template <class T, std::size_t N, std::size_t I = 0>
consteval std::array<T, N> bit_cast_array(
        std::array<T, N> vec,
        std::string_view const str,
        std::string_view const tag)
{
    if constexpr (I == N) {
        return vec;
    } else {
        std::array<char, sizeof(T) / sizeof(char)> chars = {};
        for (std::size_t j = 0; j < sizeof(T) / sizeof(char); ++j) {
            chars[j] = str[sizeof(T) / sizeof(char) * I + j];
        }

        vec[I] = std::bit_cast<T>(
                chars); // We rely on std::bit_cast because std::reinterprest_cast is not constexpr
        return bit_cast_array<T, N, I + 1>(vec, str, tag);
    }
}

template <class T, std::size_t N, std::size_t I = 0>
consteval std::array<T, N> bit_cast_array(std::string_view const str, std::string_view const tag)
{
    std::array<T, N> vec {};
    return bit_cast_array<T, N>(vec, str, tag);
}

template <class T, std::size_t N, class Ids>
struct BitCastArrayOfArrays;

template <class T, std::size_t N, std::size_t... I>
struct BitCastArrayOfArrays<T, N, std::index_sequence<I...>>
{
    static consteval std::array<std::array<T, N>, sizeof...(I)> run(
            std::array<std::string_view, sizeof...(I)> const str,
            std::string_view const tag)
    {
        return std::array<std::array<T, N>, sizeof...(I)> {bit_cast_array<T, N>(str[I], tag)...};
    }
};

} // namespace detail

template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
consteval auto YoungTableau<Dimension, TableauSeq>::load_irrep()
{
    static constexpr std::string_view str_u_coalesc_idx(detail::load_irrep_line_for_tag<0>(s_tag));
    static constexpr std::array<std::string_view, s_r> str_u_idx(
            detail::LoadIrrepIdxForTag<std::make_index_sequence<s_r>, 1>::run(s_tag));
    static constexpr std::string_view str_u_values(detail::load_irrep_line_for_tag<s_r + 1>(s_tag));
    static constexpr std::string_view str_v_coalesc_idx(
            detail::load_irrep_line_for_tag<s_r + 2>(s_tag));
    static constexpr std::array<std::string_view, s_r> str_v_idx(
            detail::LoadIrrepIdxForTag<std::make_index_sequence<s_r>, s_r + 3>::run(s_tag));
    static constexpr std::string_view str_v_values(
            detail::load_irrep_line_for_tag<2 * s_r + 3>(s_tag));

    if constexpr (str_u_values.size() != 0) {
        static constexpr std::array u_coalesc_idx = detail::bit_cast_array<
                std::size_t,
                str_u_coalesc_idx.size() / sizeof(std::size_t)>(str_u_coalesc_idx, s_tag);
        static constexpr std::array u_idx = detail::BitCastArrayOfArrays<
                std::size_t,
                str_u_idx[0].size() / sizeof(std::size_t),
                std::make_index_sequence<s_r>>::run(str_u_idx, s_tag);
        static constexpr std::array u_values = detail::
                bit_cast_array<double, str_u_values.size() / sizeof(double)>(str_u_values, s_tag);
        static constexpr std::array v_coalesc_idx = detail::bit_cast_array<
                std::size_t,
                str_v_coalesc_idx.size() / sizeof(std::size_t)>(str_v_coalesc_idx, s_tag);
        static constexpr std::array v_idx = detail::BitCastArrayOfArrays<
                std::size_t,
                str_v_idx[0].size() / sizeof(std::size_t),
                std::make_index_sequence<s_r>>::run(str_v_idx, s_tag);
        static constexpr std::array v_values = detail::
                bit_cast_array<double, str_v_values.size() / sizeof(double)>(str_v_values, s_tag);
        return std::make_pair(
                std::make_tuple(u_coalesc_idx, u_idx, u_values),
                std::make_tuple(v_coalesc_idx, v_idx, v_values));
    } else {
        return std::make_pair(
                std::make_tuple(
                        std::array<std::size_t, 1> {0},
                        std::array<std::array<std::size_t, 0>, s_r> {},
                        std::array<double, 0> {}),
                std::make_tuple(
                        std::array<std::size_t, 1> {0},
                        std::array<std::array<std::size_t, 0>, s_r> {},
                        std::array<double, 0> {}));
    }
}

namespace detail {

// Produce tag as a string (std::string features are limited at compile-time that's why we manipulate char arrays)
template <class Row>
struct YoungTableauRowToArray;

template <std::size_t... RowElement>
struct YoungTableauRowToArray<std::index_sequence<RowElement...>>
{
    static constexpr auto run()
    {
        static constexpr std::array row = {RowElement...};
        return row;
    }
};

template <class TableauSeq>
struct YoungTableauToArray;

template <class... Row>
struct YoungTableauToArray<YoungTableauSeq<Row...>>
{
    static constexpr auto run()
    {
        static constexpr std::tuple tableau = {YoungTableauRowToArray<Row>::run()...};
        return tableau;
    }
};

template <bool RowDelimiter>
struct RowToString
{
    template <std::size_t N>
    static constexpr std::array<char, 2 * N - !RowDelimiter> run(
            const std::array<std::size_t, N>& array)
    {
        std::array<char, 2 * N - !RowDelimiter> buf = {};

        std::size_t current_pos = 0;
        for (std::size_t i = 0; i < N; ++i) {
            char temp[20];
            auto [ptr, ec] = std::to_chars(temp, temp + sizeof(temp), array[i]);
            if (ec != std::errc()) {
                throw std::runtime_error("Failed to convert number to string");
            }

            for (char* p = temp; p < ptr; ++p) {
                buf[current_pos++] = *p;
            }

            if (i < N - 1) {
                buf[current_pos++] = '_';
            }
        }

        if constexpr (RowDelimiter) {
            buf[2 * N - 1] = 'l';
        }

        return buf;
    }
};

template <std::size_t... sizes>
constexpr auto concatenate(const std::array<char, sizes>&... arrays)
{
    std::array<char, (sizes + ...)> result;
    std::size_t index {};

    ((std::copy_n(arrays.begin(), sizes, result.begin() + index), index += sizes), ...);

    return result;
}

template <class RowIdx>
struct ArrayToString;

template <std::size_t... RowId>
struct ArrayToString<std::index_sequence<RowId...>>
{
    template <class Tuple>
    static constexpr auto run(Tuple const tableau)
    {
        return concatenate(
                RowToString<RowId != sizeof...(RowId) - 1>::run(std::get<RowId>(tableau))...);
    }
};

template <std::size_t size>
constexpr auto add_dimension(const std::array<char, size>& array, std::size_t d)
{
    std::array<char, size + 2> result;
    char temp[1];
    std::to_chars(temp, temp + 1, d);
    std::copy_n(array.begin(), size, result.begin());
    result[size] = 'd';
    result[size + 1] = temp[0];

    return result;
}

} // namespace detail

template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
constexpr std::string YoungTableau<Dimension, TableauSeq>::generate_tag()
{
    static constexpr std::tuple tableau = detail::YoungTableauToArray<tableau_seq>::run();
    constexpr auto row_str_wo_dimension
            = detail::ArrayToString<std::make_index_sequence<tableau_seq::shape::size()>>::run(
                    tableau);
    constexpr auto row_str = detail::add_dimension(row_str_wo_dimension, s_d);
    return std::string(row_str.begin(), row_str.end());
}

namespace detail {

// Print Young tableau in a string
template <class TableauSeq>
struct PrintYoungTableauSeq;

template <std::size_t HeadRowHeadElement, std::size_t... HeadRowTailElement, class... TailRow>
struct PrintYoungTableauSeq<
        YoungTableauSeq<std::index_sequence<HeadRowHeadElement, HeadRowTailElement...>, TailRow...>>
{
    static std::string run(std::string str)
    {
        str += std::to_string(HeadRowHeadElement) + " ";
        if constexpr (sizeof...(TailRow) == 0 && sizeof...(HeadRowTailElement) == 0) {
        } else if constexpr (sizeof...(HeadRowTailElement) == 0) {
            str += "\n";
            str = PrintYoungTableauSeq<YoungTableauSeq<TailRow...>>::run(str);
        } else {
            str = PrintYoungTableauSeq<
                    YoungTableauSeq<std::index_sequence<HeadRowTailElement...>, TailRow...>>::
                    run(str);
        }
        return str;
    }
};

template <>
struct PrintYoungTableauSeq<YoungTableauSeq<>>
{
    static std::string run(std::string str)
    {
        return str;
    }
};

} // namespace detail

template <std::size_t Dimension, misc::Specialization<YoungTableauSeq> TableauSeq>
std::ostream& operator<<(std::ostream& os, YoungTableau<Dimension, TableauSeq> const& tableau)
{
    std::string str = "";
    os << detail::PrintYoungTableauSeq<TableauSeq>::run(str);
    return os;
}

} // namespace young_tableau

} // namespace sil
