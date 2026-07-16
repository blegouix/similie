# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

add_library(similie_onelab_interface INTERFACE)

target_link_libraries(similie_onelab_interface
    INTERFACE
        DDC::core
        sil::sil
)

target_include_directories(similie_onelab_interface
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(similie_onelab main.cpp)

target_link_libraries(similie_onelab
    PRIVATE
        similie_onelab_interface
)
