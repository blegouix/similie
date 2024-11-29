# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("tensor" INTERFACE)

target_include_directories("tensor" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
target_link_libraries("tensor" INTERFACE DDC::DDC sil::young_tableau sil::misc)
else()
target_link_libraries("tensor" INTERFACE DDC::DDC sil::misc)
endif()

add_library("sil::tensor" ALIAS "tensor")
