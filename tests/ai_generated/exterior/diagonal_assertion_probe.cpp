// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#if defined(SIMILIE_TEST_DEBUG_ASSERTIONS)
#undef NDEBUG
#endif

#include <string>

#include "local_staging_fixture.hpp"

template <template <class...> class Coefficients>
int probe(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    ddc::ScopeGuard ddc(argc, argv);
    using namespace local_staging_test;
    bool const skew = argc > 2 && std::string(argv[2]) == "skew";
    bool const reduce = argc > 1 && std::string(argv[1]) == "reduction";
    auto positions = allocate<PositionIndex>(halo());
    Field<PositionIndex> position(positions);
    fill_geometry(position, skew);
    if (argc > 2 && std::string(argv[2]) == "isotropic") {
        ddc::parallel_for_each(
                Exec(),
                halo(),
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(0))
                            = 0.75 * element.uid<GridX>();
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(1))
                            = 0.75 * element.uid<GridY>();
                    position.mem(element, ddc::DiscreteElement<PositionIndex>(2))
                            = 0.75 * element.uid<GridZ>();
                });
    }
    using Indices = ddc::TypeSeq<One>;
    if (argc > 1 && std::string(argv[1]) == "weighted-hodge") {
        using Target = sil::tensor::TensorAntisymmetricIndex<
                sil::tensor::Covariant<Other>,
                sil::tensor::Covariant<Third>>;
        using Weighted = sil::exterior::WeightedLeviCivitaDiscreteHodgeStar<
                sil::exterior::CellComplex::Primal,
                ddc::TypeSeq<sil::tensor::Contravariant<Direction>>,
                ddc::TypeSeq<sil::tensor::Covariant<Other>, sil::tensor::Covariant<Third>>,
                Field<MetricIndex>,
                Field<PositionIndex>,
                Domain::discrete_element_type>;
        auto metrics = allocate<MetricIndex>(domain());
        auto inputs = allocate<One>(domain());
        auto outputs = allocate<Target>(domain());
        Field<MetricIndex> metric(metrics);
        Field<One> input(inputs);
        Field<Target> output(outputs);
        fill_form(input);
        if (argc > 3 && std::string(argv[3]) == "direct") {
            ddc::parallel_for_each(
                    Exec(),
                    domain(),
                    KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                        Weighted::
                        operator()(output[element], input[element], metric, position, element);
                    });
        } else {
            sil::exterior::StagedOperator<Weighted> const cached(Exec(), metric, position);
            cached(output, input);
        }
        Exec().fence();
        return 0;
    }
    if (argc > 3 && std::string(argv[3]) == "direct") {
        auto inputs = allocate<One>(domain());
        auto outputs = allocate<One>(domain());
        Field<One> input(inputs), output(outputs);
        fill_form(input);
        ddc::parallel_for_each(
                Exec(),
                domain(),
                KOKKOS_LAMBDA(Domain::discrete_element_type element) {
                    if (reduce)
                        sil::exterior::Reduction<
                                Indices,
                                Field<PositionIndex>,
                                Domain::discrete_element_type,
                                sil::exterior::CellComplex::Primal,
                                Coefficients>::
                        operator()(output[element], input[element], position, element);
                    else
                        sil::exterior::Reconstruction<
                                Indices,
                                Field<PositionIndex>,
                                Domain::discrete_element_type,
                                sil::exterior::CellComplex::Primal,
                                Coefficients>::
                        operator()(output[element], input[element], position, element);
                });
        Exec().fence();
    } else if (reduce) {
        sil::exterior::StagedOperator<sil::exterior::Reduction<
                Indices,
                Field<PositionIndex>,
                typename Field<One>::non_indices_domain_t::discrete_element_type,
                sil::exterior::CellComplex::Primal,
                Coefficients>>
                cached(Exec(), position, domain());
        Exec().fence();
    } else {
        sil::exterior::StagedOperator<sil::exterior::Reconstruction<
                Indices,
                Field<PositionIndex>,
                typename Field<One>::non_indices_domain_t::discrete_element_type,
                sil::exterior::CellComplex::Primal,
                Coefficients>>
                cached(Exec(), position, domain());
        Exec().fence();
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 4 && std::string(argv[4]) == "scaling")
        return probe<sil::tensor::TensorScalingIndex>(argc, argv);
    return probe<sil::tensor::TensorDiagonalIndex>(argc, argv);
}
