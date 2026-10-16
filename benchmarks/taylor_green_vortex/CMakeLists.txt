# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

set(TGV_IDEAL_GAS_HEADER "${CMAKE_CURRENT_BINARY_DIR}/taylor_green_ideal_gas.hpp")
add_custom_command(
    OUTPUT "${TGV_IDEAL_GAS_HEADER}"
    COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/ideal_gas.py"
            --output "${TGV_IDEAL_GAS_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ideal_gas.py"
            "${PROJECT_SOURCE_DIR}/src/similie/physics/fluid/navier_stokes_fourier.py"
    COMMENT "Generating Taylor-Green ideal-gas closure from SymPy"
    VERBATIM
)
# Compile each policy combination separately to bound CUDA compiler memory.
foreach(mode RANGE 0 19)
    add_library(tgv_laplacian_mode_${mode} OBJECT laplacian_mode.cpp)
    target_compile_definitions(tgv_laplacian_mode_${mode} PRIVATE TGV_LAPLACIAN_MODE=${mode})
    target_link_libraries(tgv_laplacian_mode_${mode} PRIVATE sil::exterior)
    list(APPEND TGV_LAPLACIAN_OBJECTS $<TARGET_OBJECTS:tgv_laplacian_mode_${mode}>)
endforeach()
add_executable(taylor_green_vortex taylor_green_vortex.cpp laplacian_modes.cpp
    "${TGV_IDEAL_GAS_HEADER}" ${TGV_LAPLACIAN_OBJECTS})
target_include_directories(taylor_green_vortex PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(taylor_green_vortex PRIVATE sil::sil benchmark::benchmark HDF5::HDF5 Ginkgo::ginkgo)
if(BUILD_TESTING)
    foreach(fluid incompressible compressible)
        foreach(coupling hd ehd mhd fmhd)
            foreach(transport ideal resistive)
                if(coupling STREQUAL "hd" AND transport STREQUAL "resistive")
                    continue()
                endif()
                set(validation "tgv_${fluid}_${transport}_${coupling}_validation")
                add_test(NAME "${validation}" COMMAND taylor_green_vortex --N=8
                    "--${fluid}" "--${transport}" "--${coupling}" --validate)
                set_tests_properties("${validation}" PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
            endforeach()
        endforeach()
    endforeach()
endif()
