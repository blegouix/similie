# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

if(BUILD_TESTING)
    include(GoogleTest)
    find_package(Ginkgo CONFIG REQUIRED)
    add_executable(unit_tests_solver_initial_guess initial_guess.cpp ../../main.cpp)
    target_link_libraries(unit_tests_solver_initial_guess PRIVATE GTest::gtest sil::sil Ginkgo::ginkgo)
    gtest_discover_tests(unit_tests_solver_initial_guess DISCOVERY_MODE PRE_TEST)
    foreach(solver minres bicgstab)
        add_test(NAME "SolverInitialGuess.${solver}"
            COMMAND unit_tests_solver_initial_guess --gtest_filter=SolverInitialGuess.*)
        set_tests_properties("SolverInitialGuess.${solver}" PROPERTIES
            ENVIRONMENT "SIMILIE_SOLVER=${solver};SIMILIE_SOLVER_PROGRESS_STRIDE=0")
    endforeach()
endif()
