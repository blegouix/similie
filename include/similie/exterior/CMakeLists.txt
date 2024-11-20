# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("exterior" INTERFACE)

target_include_directories("exterior" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("exterior" INTERFACE DDC::DDC sil::tensor)

add_library("sil::exterior" ALIAS "exterior")
