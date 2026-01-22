# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library("csr" INTERFACE)

target_include_directories("csr" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("csr" INTERFACE DDC::core sil::tensor)

add_library("sil::csr" ALIAS "csr")
