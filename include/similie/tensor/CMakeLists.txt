# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("tensor" INTERFACE)

target_include_directories("tensor" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Boost REQUIRED)

target_link_libraries("tensor" INTERFACE DDC::DDC Boost::boost sil::young_tableau sil::misc)

add_library("sil::tensor" ALIAS "tensor")
