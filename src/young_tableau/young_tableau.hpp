// SPDX-License-Identifier: GPL-3.0

#pragma once

#include <fstream>

namespace sil {

namespace young_tableau {

namespace detail {

template <class T, T... Args>
constexpr T sum(std::integer_sequence<T, Args...> = {})
{
    return (Args + ...);
}

template <class Mu, class Shape>
struct ConjugateIterator;

template <std::size_t... MuRowSize, std::size_t HeadRowSize, std::size_t... TailRowSize>
struct ConjugateIterator<
        std::index_sequence<MuRowSize...>,
        std::index_sequence<HeadRowSize, TailRowSize...>>
{
    using type = typename std::conditional_t<
            sizeof...(TailRowSize) == 0,
            std::index_sequence<MuRowSize..., 0>,
            typename ConjugateIterator<
                    std::index_sequence<MuRowSize...>,
                    std::index_sequence<TailRowSize...>>::type>;
};

template <std::size_t... MuRowSize>
struct ConjugateIterator<std::index_sequence<MuRowSize...>, std::index_sequence<>>
{
    using type = std::index_sequence<MuRowSize...>;
};

} // namespace detail

template <class... Row>
class YoungTableauSeq
{
public:
    using shape = std::index_sequence<Row::size()...>;

    static constexpr std::size_t rank = detail::sum(shape());

    using conjugate = detail::ConjugateIterator<std::index_sequence<shape::size()>, shape>::type;

    // using mu = YoungTableauSeq<std::index_sequence<shape::size()>>;
};

/*
def __conjugate(nu):
    l = len(nu) shape::size()
    mu = [l] YoungTableauSeq<std::index_sequence<shape::size()>>
    if l >= 1:
        for i in range(1, nu[0]):
            mu.append(0)
            j = 0
            while j < len(nu) and nu[j] > i:
                mu[i] = mu[i] + 1
                j = j + 1
    return mu
*/

namespace detail {

// Extract Row from a YoungTableauSeq at index I
template <std::size_t I, std::size_t Id, class TableauSeq>
struct ExtractRow;

template <std::size_t I, std::size_t Id, class HeadRow, class... TailRow>
struct ExtractRow<I, Id, YoungTableauSeq<HeadRow, TailRow...>>
{
    using type = std::
            conditional_t<I == Id, HeadRow, ExtractRow<I, Id + 1, YoungTableauSeq<TailRow...>>>;
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
            ExtractElem<J, Id + 1, std::index_sequence<TailElem...>>>;
};

template <std::size_t J, class Row>
using extract_elem_t = ExtractElem<J, 0, Row>::type;

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
    using shape = TableauSeq::shape;

    static constexpr std::size_t s_d = Dimension;
    static constexpr std::size_t s_r = TableauSeq::rank;

public:
    YoungTableau()
    {
        std::cout << "\033[1;31mThe representations dictionnary does not contain any "
                     "representation for the Young tableau:\033[0m\n"
                  << print_young_tableau_seq<TableauSeq>()
                  << "\n\033[1;31min dimension " + std::to_string(s_d)
                             + ". Please compile with BUILD_COMPUTE_REPRESENTATION=ON and "
                               "rerun.\033[0m\n";
        std::cout << std::to_string(s_r);
        std::cout << std::to_string(tableau_seq::conjugate::size());
    }
};

} // namespace young_tableau

} // namespace sil
