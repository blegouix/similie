# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

add_executable(taylor_green_vortex taylor_green_vortex.cpp)
target_link_libraries(taylor_green_vortex PRIVATE sil::sil benchmark::benchmark HDF5::HDF5 Ginkgo::ginkgo)
if(BUILD_TESTING)
    add_test(NAME tgv_incompressible_validation COMMAND taylor_green_vortex --N=8 --physics=incompressible --validate)
    add_test(NAME tgv_compressible_validation COMMAND taylor_green_vortex --N=8 --physics=compressible --validate)
    set_tests_properties(tgv_incompressible_validation tgv_compressible_validation
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy, h5py"
        RESULT_VARIABLE TGV_POST_PROCESS_DEPENDENCIES
        OUTPUT_QUIET ERROR_QUIET
    )
    if(TGV_POST_PROCESS_DEPENDENCIES EQUAL 0)
        add_test(NAME tgv_post_process_validation
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_post_process.py")
        set_tests_properties(tgv_post_process_validation PROPERTIES
            ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    else()
        message(STATUS "Taylor-Green post-processing tests need Python numpy and h5py")
    endif()
endif()
