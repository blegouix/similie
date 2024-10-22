// SPDX-FileCopyrightText: 2024 Baptiste Legouix
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <ddc/ddc.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include "csr.hpp"
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

// Type of index used by projectors or symmetrizers
template <class Parent>
struct declare_deriv : Parent
{
};

// Type of index used by sil::tensor::OrthonormalBasisSubspaceEigenvalueOne
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
    struct type : sil::tensor::TensorNaturalIndex<Dummy<Id>...>
    {
    };
};

template <class NaturalIds, class RankIds>
struct DummyIndex;

template <std::size_t... Id, std::size_t... RankId>
struct DummyIndex<std::index_sequence<Id...>, std::index_sequence<RankId...>>
{
    using type = sil::tensor::FullTensorIndex<
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
sil::tensor::Tensor<ElementType, ddc::DiscreteDomain<Id...>, LayoutStridedPolicy, MemorySpace>
orthogonalize(
        sil::tensor::
                Tensor<ElementType, ddc::DiscreteDomain<Id...>, LayoutStridedPolicy, MemorySpace>
                        tensor,
        sil::csr::Csr<BasisId, Id...> basis,
        std::size_t max_basis_id)
{
    ddc::Chunk eigentensor_alloc(
            ddc::DiscreteDomain<BasisId, Id...>(basis.domain()),
            ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<BasisId, Id...>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            eigentensor(eigentensor_alloc);
    for (ddc::DiscreteElement<BasisId> elem : ddc::DiscreteDomain<BasisId>(
                 ddc::DiscreteElement<BasisId>(0),
                 ddc::DiscreteVector<BasisId>(max_basis_id))) {
        sil::csr::csr2dense(eigentensor, basis.get(elem));

        ddc::Chunk scalar_prod_alloc(ddc::DiscreteDomain<> {}, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                scalar_prod(scalar_prod_alloc);
        sil::tensor::
                tensor_prod(scalar_prod, tensor, eigentensor[ddc::DiscreteElement<BasisId>(0)]);
        ddc::Chunk norm_squared_alloc(ddc::DiscreteDomain<> {}, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                norm_squared(norm_squared_alloc);
        sil::tensor::tensor_prod(norm_squared, eigentensor, eigentensor);

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

// Dummy tag used by OrthonormalBasisSubspaceEigenvalueOne (coalescent dimension of the Csr storage)
struct BasisId
{
};

} // namespace detail

} // namespace young_tableau

namespace tensor {

namespace detail {

template <>
struct IsTensorIndex<sil::young_tableau::detail::BasisId>
{
    using type = std::false_type; // TODO REMOVE
};

} // namespace detail

} // namespace tensor

namespace young_tableau {

namespace detail {

template <class Ids>
struct OrthonormalBasisSubspaceEigenvalueOne;

template <class... Id>
struct OrthonormalBasisSubspaceEigenvalueOne<sil::tensor::FullTensorIndex<Id...>>
{
    template <class YoungTableau>
    static std::pair<sil::csr::Csr<BasisId, Id...>, sil::csr::Csr<BasisId, Id...>> run(
            YoungTableau tableau)
    {
        auto [proj_alloc, proj] = tableau.template projector<Id...>();

        sil::tensor::TensorAccessor<Id...> candidate_accessor;
        ddc::DiscreteDomain<Id...> candidate_dom = candidate_accessor.mem_domain();
        ddc::Chunk candidate_alloc(candidate_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                candidate(candidate_alloc);
        ddc::Chunk prod_alloc(
                sil::tensor::tensor_prod_domain(proj, candidate),
                ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                sil::tensor::tensor_prod_domain_t<
                        typename YoungTableau::template projector_domain<Id...>,
                        ddc::DiscreteDomain<Id...>>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                prod(prod_alloc);

        ddc::DiscreteDomain<BasisId, Id...> basis_dom(
                ddc::DiscreteDomain<BasisId>(
                        ddc::DiscreteElement<BasisId>(0),
                        ddc::DiscreteVector<BasisId>(tableau.irrep_dim())),
                candidate_dom);
        sil::csr::Csr<BasisId, Id...> u(basis_dom);
        sil::csr::Csr<BasisId, Id...> v(basis_dom);
        std::size_t n_irreps = 0;
        std::size_t index = 0;
        while (n_irreps < tableau.irrep_dim()) {
            index++;
            std::vector<bool> hamming_weight_code
                    = index_hamming_weight_code(index, (Id::size() * ...));
            ddc::parallel_for_each(candidate.domain(), [&](ddc::DiscreteElement<Id...> elem) {
                candidate(elem) = hamming_weight_code[(
                        (sil::tensor::detail::stride<Id, Id...>() * elem.template uid<Id>())
                        + ...)];
            });

            sil::tensor::tensor_prod(prod, proj, candidate);
            Kokkos::deep_copy(
                    candidate.allocation_kokkos_view(),
                    prod.allocation_kokkos_view()); // We rely on Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions

            orthogonalize(candidate, v, n_irreps);

            if (ddc ::transform_reduce(
                        candidate.domain(),
                        false,
                        ddc::reducer::lor<bool>(),
                        [&](ddc::DiscreteElement<Id...> elem) { return candidate(elem) > 0.25; })) {
                ddc::Chunk
                        norm_squared_alloc(ddc::DiscreteDomain<> {}, ddc::HostAllocator<double>());
                sil::tensor::Tensor<
                        double,
                        ddc::DiscreteDomain<>,
                        std::experimental::layout_right,
                        Kokkos::DefaultHostExecutionSpace::memory_space>
                        norm_squared(norm_squared_alloc);
                sil::tensor::tensor_prod(norm_squared, candidate, candidate);
                ddc::parallel_for_each(candidate.domain(), [&](ddc::DiscreteElement<Id...> elem) {
                    candidate(elem) /= Kokkos::sqrt(norm_squared(ddc::DiscreteElement<>()));
                });
                // Not sure if u = v is correct in any case (ie. complex tensors ?)
                u.push_back(candidate);
                v.push_back(candidate);
                n_irreps++;
            }
        }
        return std::pair<sil::csr::Csr<BasisId, Id...>, sil::csr::Csr<BasisId, Id...>>(u, v);
    }
};

// Load binary files and build u and v static constexpr Csr at compile-time
consteval std::string_view load_irrep_as_string(std::string_view const irrep_tag)
{
    constexpr static char raw[] = {
#embed IRREPS_DICT_PATH
    };

    constexpr static std::string_view str(raw, sizeof(raw));

    size_t tagPos = str.find(irrep_tag);

    if (tagPos != std::string::npos) {
        std::size_t endOfTagLine = str.find('\n', tagPos);
        if (endOfTagLine != std::string::npos) {
            std::size_t nextLineStart = endOfTagLine + 1;
            std::size_t nextLineEnd = str.find('\n', nextLineStart);

            return std::string_view(str.data() + nextLineStart, nextLineEnd - nextLineStart);
        }
    }

    return "";
}

template <std::size_t N, std::size_t I = 0>
consteval std::array<double, N> bit_cast_char_array_to_double_vector(
        std::array<double, N> vec,
        std::string_view const str,
        std::string_view const irrep_tag)
{
    if constexpr (I == N) {
        return vec;
    } else {
        std::array<char, sizeof(double) / sizeof(char)> chars = {};
        for (std::size_t j = 0; j < sizeof(double) / sizeof(char); ++j) {
            chars[j] = str[sizeof(double) / sizeof(char) * I + j];
        }

        vec[I] = std::bit_cast<double>(
                chars); // We rely on std::bit_cast because std::reinterprest_cast is not constexpr
        return bit_cast_char_array_to_double_vector<N, I + 1>(vec, str, irrep_tag);
    }
}

template <std::size_t N, std::size_t I = 0>
consteval std::array<double, N> bit_cast_char_array_to_double_vector(
        std::string_view const str,
        std::string_view const irrep_tag)
{
    std::array<double, N> vec {};
    return bit_cast_char_array_to_double_vector<N>(vec, str, irrep_tag);
}

} // namespace detail

/**
 * YoungTableau class
 */
template <std::size_t Dimension, class TableauSeq>
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

    static constexpr std::string_view s_tag = "tag";

    static consteval auto build_u_values()
    {
        static constexpr std::string_view str(detail::load_irrep_as_string(s_tag));

        if constexpr (str.size() != 0) {
            return detail::bit_cast_char_array_to_double_vector<
                    str.size() / sizeof(double),
                    0>(str, s_tag);
        } else {
            return std::array<double, 0> {};
        }
    }

    static constexpr std::array s_u_values = build_u_values();

public:
    YoungTableau()
    {
        auto [u, v] = detail::OrthonormalBasisSubspaceEigenvalueOne<
                detail::dummy_index_t<s_d, s_r>>::run(*this);
        u.write(IRREPS_DICT_PATH, "tag");
    }

    static consteval std::size_t dimension()
    {
        return s_d;
    }

    static consteval std::size_t rank()
    {
        return s_r;
    }

    static consteval std::size_t irrep_dim()
    {
        return s_irrep_dim;
    }

    template <class... Id>
    using projector_domain = ddc::DiscreteDomain<detail::declare_deriv<Id>..., Id...>;

    template <class... Id>
    static auto projector();

    void print_representation_absent(); // TODO REMOVE

    void print_u() const
    {
        for (std::size_t i = 0; i < s_u_values.size(); ++i) {
            std::cout << s_u_values[i] << "\n";
        }
    }
};

namespace detail {

// Build index for symmetrizer (such that sym*proj is properly defined)
template <class OId, class... Id>
using symmetrizer_index_t = std::conditional_t<
        (ddc::type_seq_rank_v<OId, ddc::detail::TypeSeq<Id...>> < (sizeof...(Id) / 2)),
        declare_deriv<OId>,
        ddc::type_seq_element_t<
                static_cast<std::size_t>(
                        std::
                                max(static_cast<std::ptrdiff_t>(0),
                                    static_cast<std::ptrdiff_t>(
                                            ddc::type_seq_rank_v<
                                                    OId,
                                                    ddc::detail::TypeSeq<
                                                            Id...>> - (sizeof...(Id) / 2)))),
                ddc::detail::TypeSeq<Id...>>>;

// Lambda to fill identity or transpose projectors
template <std::size_t Dimension, std::size_t Rank, class... NaturalId>
std::function<
        void(sil::tensor::Tensor<
                     double,
                     ddc::DiscreteDomain<NaturalId...>,
                     std::experimental::layout_right,
                     Kokkos::DefaultHostExecutionSpace::memory_space>,
             ddc::DiscreteElement<NaturalId...>)>
tr_lambda(std::array<std::size_t, Rank> idx_to_permute)
{
    return [=](sil::tensor::Tensor<
                       double,
                       ddc::DiscreteDomain<NaturalId...>,
                       std::experimental::layout_right,
                       Kokkos::DefaultHostExecutionSpace::memory_space> t,
               ddc::DiscreteElement<NaturalId...> elem) {
        std::array<std::size_t, sizeof...(NaturalId)> elem_array = ddc::detail::array(elem);
        bool has_to_be_one = true;
        for (std::size_t i = 0; i < Rank; ++i) {
            has_to_be_one
                    = has_to_be_one && (elem_array[i] == elem_array[idx_to_permute[i] + Rank]);
        }
        if (has_to_be_one) {
            t(elem) = 1;
        }
    };
}

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
static sil::tensor::Tensor<
        double,
        ddc::DiscreteDomain<Id...>,
        std::experimental::layout_right,
        Kokkos::DefaultHostExecutionSpace::memory_space>
fill_symmetrizer(
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<Id...>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space> sym,
        std::array<std::size_t, sizeof...(Id) / 2> idx_to_permute)
{
    ddc::Chunk tr_alloc(sym.domain(), ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Id...>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            tr(tr_alloc);
    ddc::parallel_fill(tr, 0);
    tr.apply_lambda(tr_lambda<Dimension, sizeof...(Id) / 2, Id...>(idx_to_permute));

    if constexpr (!AntiSym) {
        tr *= 1. / boost::math::factorial<double>(sizeof...(Id) / 2);
    } else {
        tr *= permutation_parity(idx_to_permute)
              / boost::math::factorial<double>(sizeof...(Id) / 2);
    }
    sym += tr;

    return sym;
}

/*
 Compute all permutations on a subset of indexes in idx_to_permute, keep the
 rest of the indexes where they are.
 */
template <std::size_t Nt, std::size_t Ns>
static std::vector<std::array<std::size_t, Nt>> permutations_subset(
        std::array<std::size_t, Nt> t,
        std::array<std::size_t, Ns> subset_values)
{
    std::array<std::size_t, Ns> subset_indexes;
    std::array<std::size_t, Ns> elements_to_permute;
    int j = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (std::find(subset_values.begin(), subset_values.end(), t[i] + 1)
            != std::end(subset_values)) {
            subset_indexes[j] = i;
            elements_to_permute[j++] = t[i];
        }
    }

    std::vector<std::array<std::size_t, Nt>> result;
    do {
        std::array<std::size_t, Nt> tmp = t;
        for (std::size_t i = 0; i < Ns; ++i) {
            tmp[subset_indexes[i]] = elements_to_permute[i];
        }
        result.push_back(tmp);
    } while (std::next_permutation(elements_to_permute.begin(), elements_to_permute.end()));

    return result;
}

// Compute projector
template <class PartialTableauSeq, std::size_t Dimension, bool AntiSym = 0>
struct Projector;

template <std::size_t... ElemOfHeadRow, class... TailRow, std::size_t Dimension, bool AntiSym>
struct Projector<
        YoungTableauSeq<std::index_sequence<ElemOfHeadRow...>, TailRow...>,
        Dimension,
        AntiSym>
{
    template <class... Id>
    static sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Id...>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
    run(sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<Id...>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space> proj)
    {
        if constexpr (sizeof...(ElemOfHeadRow) >= 2) {
            // Allocate & build a symmetric projector for the row
            sil::tensor::TensorAccessor<symmetrizer_index_t<Id, Id...>...> sym_accessor;
            ddc::DiscreteDomain<symmetrizer_index_t<Id, Id...>...> sym_dom
                    = sym_accessor.mem_domain();

            ddc::Chunk sym_alloc(sym_dom, ddc::HostAllocator<double>());
            sil::tensor::Tensor<
                    double,
                    ddc::DiscreteDomain<symmetrizer_index_t<Id, Id...>...>,
                    std::experimental::layout_right,
                    Kokkos::DefaultHostExecutionSpace::memory_space>
                    sym(sym_alloc);
            ddc::parallel_fill(sym, 0);
            std::array<std::size_t, sizeof...(Id) / 2> idx_to_permute;
            for (std::size_t i = 0; i < sizeof...(Id) / 2; ++i) {
                idx_to_permute[i] = i;
            }
            std::array<std::size_t, sizeof...(ElemOfHeadRow)> row_values {ElemOfHeadRow...};
            auto idx_permutations = detail::permutations_subset(idx_to_permute, row_values);
            for (int i = 0; i < idx_permutations.size(); ++i) {
                fill_symmetrizer<
                        Dimension,
                        AntiSym,
                        symmetrizer_index_t<Id, Id...>...>(sym, idx_permutations[i]);
            }

            // Extract the symmetric part (for the row) of the projector (requires an intermediate prod tensor)
            ddc::Chunk prod_alloc(tensor_prod_domain(sym, proj), ddc::HostAllocator<double>());
            sil::tensor::Tensor<
                    double,
                    sil::tensor::tensor_prod_domain_t<
                            ddc::DiscreteDomain<symmetrizer_index_t<Id, Id...>...>,
                            ddc::DiscreteDomain<Id...>>,
                    std::experimental::layout_right,
                    Kokkos::DefaultHostExecutionSpace::memory_space>
                    prod(prod_alloc);
            sil::tensor::tensor_prod(prod, sym, proj);
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

template <std::size_t Dimension, class TableauSeq>
template <class... Id>
auto YoungTableau<Dimension, TableauSeq>::projector()
{
    static_assert(sizeof...(Id) == s_r);
    sil::tensor::TensorAccessor<detail::declare_deriv<Id>..., Id...> proj_accessor;
    ddc::DiscreteDomain<detail::declare_deriv<Id>..., Id...> proj_dom = proj_accessor.mem_domain();

    // Allocate a projector and fill it as an identity tensor
    ddc::Chunk proj_alloc(proj_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor<
            double,
            ddc::DiscreteDomain<detail::declare_deriv<Id>..., Id...>,
            std::experimental::layout_right,
            Kokkos::DefaultHostExecutionSpace::memory_space>
            proj(proj_alloc);
    ddc::parallel_fill(proj, 0);
    std::array<std::size_t, s_r> idx_to_permute;
    for (std::size_t i = 0; i < s_r; ++i) {
        idx_to_permute[i] = i;
    }
    proj.apply_lambda(
            detail::tr_lambda<s_d, s_r, detail::declare_deriv<Id>..., Id...>(idx_to_permute));

    // Build the projector
    detail::Projector<tableau_seq, Dimension>::run(proj);
    detail::Projector<typename dual::tableau_seq, Dimension, 1>::run(proj);
    return std::make_tuple(std::move(proj_alloc), proj);
}

template <std::size_t Dimension, class TableauSeq>
void YoungTableau<Dimension, TableauSeq>::print_representation_absent()
{
    std::cout << "\033[1;31mThe representations dictionnary does not contain any "
                 "representation for the Young tableau:\033[0m\n"
              << *this
              << "\n\033[1;31min dimension " + std::to_string(s_d)
                         + ". Please compile with BUILD_COMPUTE_REPRESENTATION=ON and "
                           "rerun.\033[0m\n";
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

template <std::size_t Dimension, class TableauSeq>
std::ostream& operator<<(std::ostream& os, YoungTableau<Dimension, TableauSeq> const& tableau)
{
    std::string str = "";
    os << detail::PrintYoungTableauSeq<TableauSeq>::run(str);
    return os;
}

} // namespace young_tableau

} // namespace sil
