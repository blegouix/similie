# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_library("young_tableau" INTERFACE)

target_include_directories("young_tableau" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Boost REQUIRED)

target_link_libraries("young_tableau" INTERFACE DDC::DDC Boost::boost sil::tensor sil::csr)

add_library("sil::young_tableau" ALIAS "young_tableau")
