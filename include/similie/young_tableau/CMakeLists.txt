# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_library("young_tableau" INTERFACE)

target_include_directories("young_tableau" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("young_tableau" INTERFACE DDC::DDC sil::tensor sil::csr)

add_library("sil::young_tableau" ALIAS "young_tableau")
