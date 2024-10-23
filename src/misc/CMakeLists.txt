# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0

add_library("misc" INTERFACE)

target_include_directories("misc" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Boost REQUIRED)

target_link_libraries("misc" INTERFACE DDC::DDC)

add_library("sil::misc" ALIAS "misc")
