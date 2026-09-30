# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

add_subdirectory(csr)
add_subdirectory(exterior)
add_subdirectory(mesher)
add_subdirectory(tensor)
if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
add_subdirectory(young_tableau)
endif()

if(SIMILIE_BUILD_ONELAB_INTERFACE)
    include(GoogleTest)
    add_executable(unit_tests_elasticity physics/elasticity.cpp main.cpp)
    target_link_libraries(unit_tests_elasticity PRIVATE GTest::gtest similie_onelab_interface)
    gtest_discover_tests(unit_tests_elasticity DISCOVERY_MODE PRE_TEST)
endif()
