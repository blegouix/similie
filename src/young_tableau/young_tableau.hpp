// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <fstream>

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

// Type of index used by projectors
template <std::size_t I>
struct X
{
};

template <class Ids>
struct NaturalIndex;

template <std::size_t... Id>
struct NaturalIndex<std::index_sequence<Id...>>
{
    using type = sil::tensor::TensorNaturalIndex<X<Id>...>;
};

template <class NaturalIndex, class RankIds>
struct ProjectorIndex;

template <class NaturalIndex, std::size_t... RankId>
struct ProjectorIndex<NaturalIndex, std::index_sequence<RankId...>>
{
    using type = sil::tensor::FullTensorIndex<
            std::conditional_t<RankId == -1, NaturalIndex, NaturalIndex>...>;
};

template <std::size_t Dimension, std::size_t Rank>
using projector_index_t = ProjectorIndex<
        typename NaturalIndex<std::make_index_sequence<Dimension>>::type,
        std::make_index_sequence<2 * Rank>>::type;

// Lambda to fill identity or transpose projectors
template <std::size_t Dimension, std::size_t Rank, std::size_t N>
auto tr_lambda(std::array<std::size_t, N> idx_to_permute)
{
    auto lambda_to_return
            = [](sil::tensor::Tensor<
                         double,
                         ddc::DiscreteDomain<detail::projector_index_t<Dimension, Rank>>,
                         std::experimental::layout_right,
                         Kokkos::DefaultHostExecutionSpace::memory_space> t,
                 std::array<std::size_t, N> idx) { t[idx] = 1; };
    return lambda_to_return;
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
    using tableau_seq = TableauSeq;

public:
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
        projector();
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

    static void projector()
    {
        sil::tensor::TensorAccessor<detail::projector_index_t<s_d, s_r>> tensor_accessor;
        ddc::DiscreteDomain<detail::projector_index_t<s_d, s_r>> tensor_dom
                = tensor_accessor.mem_domain();

        ddc::Chunk id_tensor_alloc(tensor_dom, ddc::HostAllocator<double>());
        sil::tensor::Tensor<
                double,
                ddc::DiscreteDomain<detail::projector_index_t<s_d, s_r>>,
                std::experimental::layout_right,
                Kokkos::DefaultHostExecutionSpace::memory_space>
                id_tensor(id_tensor_alloc);
        ddc::parallel_fill(id_tensor, 0);

        std::cout << id_tensor.extents();
        std::cout << id_tensor(0);
    }

    static std::string print()
    {
        return print_young_tableau_seq<tableau_seq>();
    }
};

} // namespace young_tableau

} // namespace sil
