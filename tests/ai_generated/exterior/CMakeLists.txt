# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

include(GoogleTest)

add_executable(unit_tests_interior_product interior_product.cpp ../../main.cpp)
target_link_libraries(unit_tests_interior_product PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_interior_product DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_integration integration.cpp ../../main.cpp)
target_link_libraries(unit_tests_integration PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_integration DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_levi_civita_continuous_hodge_star
    levi_civita_continuous_hodge_star.cpp ../../main.cpp)
target_link_libraries(unit_tests_levi_civita_continuous_hodge_star
    PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_levi_civita_continuous_hodge_star DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_staged_laplacian staged_laplacian.cpp ../../main.cpp)
target_link_libraries(unit_tests_staged_laplacian PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_staged_laplacian DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_tgv_staged_codifferential tgv_staged_codifferential.cpp ../../main.cpp)
target_include_directories(unit_tests_tgv_staged_codifferential
    PRIVATE "${PROJECT_SOURCE_DIR}/benchmarks/taylor_green_vortex")
target_link_libraries(unit_tests_tgv_staged_codifferential PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_tgv_staged_codifferential DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_reconstruction_application reconstruction_application.cpp ../../main.cpp)
target_link_libraries(unit_tests_reconstruction_application PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_reconstruction_application DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_incidence_application incidence_application.cpp ../../main.cpp)
target_link_libraries(unit_tests_incidence_application PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_incidence_application DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_covariant_derivative covariant_derivative.cpp ../../main.cpp)
target_link_libraries(unit_tests_covariant_derivative PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_covariant_derivative DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_extrapolation extrapolation.cpp ../../main.cpp)
target_link_libraries(unit_tests_extrapolation PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_extrapolation DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_hodge_star hodge_star.cpp ../../main.cpp)

target_link_libraries(unit_tests_hodge_star
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_hodge_star DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_reduction_and_reconstruction reduction_and_reconstruction.cpp ../../main.cpp)

target_link_libraries(unit_tests_reduction_and_reconstruction
    PUBLIC
        GTest::gtest
        DDC::core
        sil::exterior
)

gtest_discover_tests(unit_tests_reduction_and_reconstruction DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_exterior_ai_generated exterior.cpp ../../main.cpp)
target_link_libraries(unit_tests_exterior_ai_generated PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_exterior_ai_generated DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_stencil_indices stencil_indices.cpp ../../main.cpp)
target_link_libraries(unit_tests_stencil_indices PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_stencil_indices DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_staged_local_operators staged_local_operators.cpp ../../main.cpp)
target_link_libraries(unit_tests_staged_local_operators PUBLIC GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_staged_local_operators DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_weighted_levi_civita_discrete_hodge_star
    weighted_levi_civita_discrete_hodge_star.cpp ../../main.cpp)
target_link_libraries(unit_tests_weighted_levi_civita_discrete_hodge_star
    PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_weighted_levi_civita_discrete_hodge_star DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_scaling scaling.cpp ../../main.cpp)
target_link_libraries(unit_tests_scaling PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_scaling DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_quadrature quadrature.cpp ../../main.cpp)
target_link_libraries(unit_tests_quadrature PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_quadrature DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_vertex_form_transfer vertex_form_transfer.cpp ../../main.cpp)
target_link_libraries(unit_tests_vertex_form_transfer PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_vertex_form_transfer DISCOVERY_MODE PRE_TEST)

add_executable(unit_tests_staged_bilinear_operators staged_bilinear_operators.cpp ../../main.cpp)
target_link_libraries(unit_tests_staged_bilinear_operators PRIVATE GTest::gtest DDC::core sil::exterior)
gtest_discover_tests(unit_tests_staged_bilinear_operators DISCOVERY_MODE PRE_TEST)
