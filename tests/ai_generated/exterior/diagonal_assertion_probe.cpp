// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#if defined(SIMILIE_TEST_DEBUG_ASSERTIONS)
#undef NDEBUG
#endif

#include <string>

#include "local_staging_fixture.hpp"

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    ddc::ScopeGuard ddc(argc, argv);
    using namespace local_staging_test;
    bool const skew = argc > 2 && std::string(argv[2]) == "skew";
    bool const reduce = argc > 1 && std::string(argv[1]) == "reduction";
    auto positions = allocate<PositionIndex>(halo());
    Field<PositionIndex> position(positions);
    fill_geometry(position, skew);
    using Indices = ddc::TypeSeq<One>;
    if (reduce) {
        sil::exterior::StagedReduction<
                Indices,
                Field<One>,
                Field<One>,
                Field<PositionIndex>,
                Exec,
                sil::exterior::CellComplex::Primal,
                sil::tensor::TensorDiagonalIndex>
                cached(Exec(), position, domain());
        Exec().fence();
    } else {
        sil::exterior::StagedReconstruction<
                Indices,
                Field<One>,
                Field<One>,
                Field<PositionIndex>,
                Exec,
                sil::exterior::CellComplex::Primal,
                sil::tensor::TensorDiagonalIndex>
                cached(Exec(), position, domain());
        Exec().fence();
    }
    return 0;
}
