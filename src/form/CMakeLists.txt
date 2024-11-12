# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("form" INTERFACE)

target_include_directories("form" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Boost REQUIRED)

target_link_libraries("form" INTERFACE DDC::DDC sil::tensor)

add_library("sil::form" ALIAS "form")
