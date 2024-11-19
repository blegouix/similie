# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("normalization" INTERFACE)

target_include_directories("normalization" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Boost REQUIRED)

target_link_libraries("normalization" INTERFACE DDC::DDC sil::tensor)

add_library("sil::normalization" ALIAS "normalization")
