# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("csr" INTERFACE)

target_include_directories("csr" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("csr" INTERFACE DDC::DDC sil::tensor)

add_library("sil::csr" ALIAS "csr")
