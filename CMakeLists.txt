# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.22)
project(SimiLie VERSION 0.0.0 LANGUAGES C CXX) # C is required because PDI::PDI_C propagates C compile features to consumers.

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

set(CMAKE_CXX_STANDARD 23 CACHE INTERNAL "The C++ standard whose features are requested to build this project.")

# List of options
option(SIMILIE_BUILD_DOCUMENTATION "Build SimiLie documentation/website" OFF)
option(SIMILIE_BUILD_ONLY_DOCUMENTATION "Build only documentation, skip all other targets and dependencies. Requires SIMILIE_BUILD_DOCUMENTATION=ON" OFF)
option(SIMILIE_BUILD_YOUNG_TABLEAU "Build module dedicated to Young tableau indexing" OFF) # Requires compiler supporting #embed directive
option(SIMILIE_BUILD_ONELAB_INTERFACE "Build the ONELAB interface executable" ON)
option(SIMILIE_DEBUG_LOG "Enable logging of SimiLie internal kernels" OFF)
option(SIMILIE_ASSERT_EXAMPLE_RESULTS_CORRECTNESS "Assert example result correctness at runtime" OFF)

# Validate SIMILIE_BUILD_ONLY_DOCUMENTATION option
if("${SIMILIE_BUILD_ONLY_DOCUMENTATION}" AND NOT "${SIMILIE_BUILD_DOCUMENTATION}")
  message(FATAL_ERROR "SIMILIE_BUILD_ONLY_DOCUMENTATION requires SIMILIE_BUILD_DOCUMENTATION=ON")
endif()

# Set default DDC options when included
option(DDC_BUILD_BENCHMARKS       "Build DDC benchmarks." OFF)
option(DDC_BUILD_DOCUMENTATION    "Build DDC documentation/website" OFF)
option(DDC_BUILD_EXAMPLES         "Build DDC examples" OFF)
option(DDC_BUILD_TESTS            "Build DDC tests if BUILD_TESTING is enabled" OFF)

# Custom variables
if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
  add_compile_definitions("BUILD_YOUNG_TABLEAU")
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/irreps_dict.bin)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/irreps_dict.bin "")
  endif()
  add_compile_definitions("IRREPS_DICT_PATH=\"${CMAKE_CURRENT_BINARY_DIR}/irreps_dict.bin\"")
endif()

# Dependencies

set(SIMILIE_DEPENDENCY_POLICIES "AUTO" "EMBEDDED" "INSTALLED" "SUBPROJECT")

## Use CTest for running tests
include(CTest)

## if tests are enabled, use googletest from `vendor/`
### we use it to write unit tests
if("${BUILD_TESTING}")
  set(SIMILIE_GTest_DEPENDENCY_POLICY "AUTO" CACHE STRING "Policy to find the `GTest` package. Options: ${SIMILIE_DEPENDENCY_POLICIES}")
  set_property(CACHE SIMILIE_GTest_DEPENDENCY_POLICY PROPERTY STRINGS ${SIMILIE_DEPENDENCY_POLICIES})

  if("${SIMILIE_GTest_DEPENDENCY_POLICY}" STREQUAL "AUTO")
    find_package(GTest "1.15" QUIET)
    if(NOT "${GTest_FOUND}")
      add_subdirectory("vendor/googletest/" "googletest") # SYSTEM)
    endif()
  elseif("${SIMILIE_GTest_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    add_subdirectory("vendor/googletest/" "googletest") # SYSTEM)
  elseif("${SIMILIE_GTest_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(GTest "1.15" REQUIRED)
  endif()
endif()

## Use the discrete domain computation library (ddc) from `vendor/`
if(Kokkos_ENABLE_CUDA)
  set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Enable constexpr calls in CUDA device code" FORCE)
endif()
if(NOT "${SIMILIE_BUILD_ONLY_DOCUMENTATION}")
  add_subdirectory("vendor/ddc/" "ddc") # SYSTEM)
endif()

if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
  set(CMAKE_CXX_STANDARD 26 CACHE INTERNAL "The C++ standard whose features are requested to build this project.")
endif()

## Doxygen
if("${SIMILIE_BUILD_DOCUMENTATION}")
    find_package(Doxygen 1.8.13 REQUIRED OPTIONAL_COMPONENTS dot)
endif()

###############################################################################################
#                              Build libraries and executables
###############################################################################################
# Our project code

if(NOT "${SIMILIE_BUILD_ONLY_DOCUMENTATION}")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_library(similie INTERFACE)
  add_library(sil::sil ALIAS similie)
  target_compile_features(similie INTERFACE cxx_std_23)
  target_include_directories(
      similie
      INTERFACE
          "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
          "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>"
          "$<INSTALL_INTERFACE:src>"
  )
  target_link_libraries(similie INTERFACE
      DDC::core
      DDC::splines
  )
  if("${SIMILIE_DEBUG_LOG}")
    target_compile_definitions(similie INTERFACE SIMILIE_DEBUG_LOG_)
  endif()

  add_subdirectory(src/similie)
  if("${SIMILIE_BUILD_ONELAB_INTERFACE}")
    add_subdirectory(onelab_interface)
  endif()
  add_subdirectory(tests)
  add_subdirectory(examples)

  set(SIMILIE_INTERNAL_TARGETS
      csr
      exterior
      mesher
      misc
      physics
      solvers
      tensor
  )
  if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
    list(APPEND SIMILIE_INTERNAL_TARGETS young_tableau)
  endif()

  if("${SIMILIE_DEBUG_LOG}")
    foreach(similie_target IN LISTS SIMILIE_INTERNAL_TARGETS)
      if(TARGET ${similie_target})
        target_compile_definitions(${similie_target} INTERFACE SIMILIE_DEBUG_LOG_)
      endif()
    endforeach()
  endif()

  target_link_libraries(similie INTERFACE ${SIMILIE_INTERNAL_TARGETS})
endif()

if("${SIMILIE_BUILD_DOCUMENTATION}")
    add_subdirectory(docs/)
endif()
