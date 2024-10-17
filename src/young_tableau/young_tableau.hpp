// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <fstream>

#include <ddc/ddc.hpp>

#include <boost/math/special_functions/factorials.hpp>

#include "tensor.hpp"

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

// Extract Elem from a row of YoungTableauSeq at index J
template <std::size_t J, std::size_t Id, class Row>
struct ExtractElem;

template <std::size_t J, std::size_t Id, std::size_t HeadElem, std::size_t... TailElem>
struct ExtractElem<J, Id, std::index_sequence<HeadElem, TailElem...>>
{
    using type = std::conditional_t<
            J == Id,
            std::index_sequence<HeadElem>,
            typename ExtractElem<J, Id + 1, std::index_sequence<TailElem...>>::type>;
};

template <std::size_t J, class Row>
using extract_elem_t = ExtractElem<J, 0, Row>::type;

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

// Helper to sum the partial contributions of hooks to get hook lengths (= hooks+hooks_of_dual^T-1)
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

// Print Young tableau in a string
template <class TableauSeq>
struct PrintYoungTableauSeq;

template <std::size_t HeadRowHeadElement, std::size_t... HeadRowTailElement, class... TailRow>
struct PrintYoungTableauSeq<
        YoungTableauSeq<std::index_sequence<HeadRowHeadElement, HeadRowTailElement...>, TailRow...>>
{
    static std::string run(std::string os)
    {
        os += std::to_string(HeadRowHeadElement) + " ";
        if constexpr (sizeof...(TailRow) == 0 && sizeof...(HeadRowTailElement) == 0) {
        } else if constexpr (sizeof...(HeadRowTailElement) == 0) {
            os += "\n";
            os = PrintYoungTableauSeq<YoungTableauSeq<TailRow...>>::run(os);
        } else {
            os = PrintYoungTableauSeq<
                    YoungTableauSeq<std::index_sequence<HeadRowTailElement...>, TailRow...>>::
                    run(os);
        }
        return os;
    }
};

template <>
struct PrintYoungTableauSeq<YoungTableauSeq<>>
{
    static std::string run(std::string os)
    {
        return os;
    }
};

} // namespace detail

template <class TableauSeq>
std::string print_young_tableau_seq()
{
    return detail::PrintYoungTableauSeq<TableauSeq>::run("");
}

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

public:
    YoungTableau()
    {
        std::cout << "\033[1;31mThe representations dictionnary does not contain any "
                     "representation for the Young tableau:\033[0m\n"
                  << print()
                  << "\n\033[1;31min dimension " + std::to_string(s_d)
                             + ". Please compile with BUILD_COMPUTE_REPRESENTATION=ON and "
                               "rerun.\033[0m\n";
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
    auto projector();

    static std::string print()
    {
        return print_young_tableau_seq<tableau_seq>();
    }
};

namespace detail {

/*
     Given a permutation of the digits 0..N, 
     returns its parity (or sign): +1 for even parity; -1 for odd.
     */
template <std::size_t Nt>
static int permutation_parity(std::array<std::size_t, Nt> lst)
{
    int parity = 1;
    for (int i = 0; i < lst.size() - 1; ++i) {
        parity *= -1;
        std::size_t mn = std::distance(lst.begin(), std::min_element(lst.begin() + i, lst.end()));
        std::swap(lst[i], lst[mn]);
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
    tr.fill_using_lambda(detail::tr_lambda<Dimension, sizeof...(Id) / 2, Id...>(idx_to_permute));

    if constexpr (!AntiSym) {
        tr *= 1. / boost::math::factorial<double>(sizeof...(Id) / 2);
    } else {
        tr *= permutation_parity(idx_to_permute) / boost::math::factorial<double>(sizeof...(Id));
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
        if (std::find(subset_values.begin(), subset_values.end(), t[i])
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
            sil::tensor::TensorAccessor<detail::symmetrizer_index_t<Id, Id...>...> sym_accessor;
            ddc::DiscreteDomain<detail::symmetrizer_index_t<Id, Id...>...> sym_dom
                    = sym_accessor.mem_domain();

            ddc::Chunk sym_alloc(sym_dom, ddc::HostAllocator<double>());
            sil::tensor::Tensor<
                    double,
                    ddc::DiscreteDomain<detail::symmetrizer_index_t<Id, Id...>...>,
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
                        detail::symmetrizer_index_t<Id, Id...>...>(sym, idx_permutations[i]);
            }

            // Extract the symmetric part (for the row) of the projector (requires an intermediate prod tensor)
            ddc::Chunk prod_alloc(tensor_prod_domain(sym, proj), ddc::HostAllocator<double>());
            sil::tensor::Tensor<
                    double,
                    sil::tensor::tensor_prod_domain_t<
                            ddc::DiscreteDomain<detail::symmetrizer_index_t<Id, Id...>...>,
                            ddc::DiscreteDomain<Id...>>,
                    std::experimental::layout_right,
                    Kokkos::DefaultHostExecutionSpace::memory_space>
                    prod(prod_alloc);
            sil::tensor::tensor_prod(prod, sym, proj);
            Kokkos::deep_copy(
                    proj.allocation_kokkos_view(),
                    prod.allocation_kokkos_view()); // We use Kokkos::deep_copy in place of ddc::parallel_deepcopy to avoid type verification of the type dimensions
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
    proj.fill_using_lambda(
            detail::tr_lambda<s_d, s_r, detail::declare_deriv<Id>..., Id...>(idx_to_permute));

    // Build the projector
    detail::Projector<tableau_seq, Dimension>::run(proj);
    detail::Projector<typename dual::tableau_seq, Dimension, 1>::run(proj);
    return std::make_tuple(std::move(proj_alloc), proj);
}

} // namespace young_tableau

} // namespace sil
