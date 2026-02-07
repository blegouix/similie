# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(SimiLie VERSION 0.0.0 LANGUAGES CXX)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

set(CMAKE_CXX_STANDARD 20 CACHE INTERNAL "The C++ standard whose features are requested to build this project.")

# List of options
option(SIMILIE_BUILD_YOUNG_TABLEAU "Build module dedicated to Young tableau indexing" OFF) # Requires compiler supporting #embed directive
option(SIMILIE_BUILD_DOCUMENTATION "Build SimiLie documentation/website" OFF)
option(SIMILIE_DEBUG_LOG "Enable logging of SimiLie internal kernels" OFF)

# Set default DDC options when included
option(DDC_BUILD_BENCHMARKS       "Build DDC benchmarks." OFF)
option(DDC_BUILD_DOCUMENTATION    "Build DDC documentation/website" OFF)
option(DDC_BUILD_EXAMPLES         "Build DDC examples" OFF)
option(DDC_BUILD_TESTS            "Build DDC tests if BUILD_TESTING is enabled" OFF)

# Custom variables
if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
  set(CMAKE_CXX_STANDARD 23 CACHE INTERNAL "The C++ standard whose features are requested to build this project.")
  add_compile_definitions("BUILD_YOUNG_TABLEAU")
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/irreps_dict.bin)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/irreps_dict.bin "")
  endif()
  add_compile_definitions("IRREPS_DICT_PATH=\"${CMAKE_CURRENT_BINARY_DIR}/irreps_dict.bin\"")
endif()

# Dependencies

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
add_subdirectory("vendor/ddc/" "ddc") # SYSTEM)

## Doxygen
if("${SIMILIE_BUILD_DOCUMENTATION}")
    find_package(Doxygen 1.8.13 REQUIRED OPTIONAL_COMPONENTS dot)
endif()

###############################################################################################
#                              Build libraries and executables
###############################################################################################
# Our project code

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(similie INTERFACE)
add_library(sil::sil ALIAS similie)
if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
  target_compile_features(similie INTERFACE cxx_std_23)
else()
  target_compile_features(similie INTERFACE cxx_std_20)
endif()
target_include_directories(
    similie
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
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
add_subdirectory(tests)
add_subdirectory(examples)
if("${SIMILIE_BUILD_DOCUMENTATION}")
    add_subdirectory(docs/)
endif()

set(SIMILIE_INTERNAL_TARGETS
    csr
    exterior
    mesher
    misc
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
