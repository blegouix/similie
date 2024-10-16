# SPDX-License-Identifier: GPL-3.0

add_library("young_tableau" INTERFACE)

target_include_directories("young_tableau" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Boost REQUIRED)

target_link_libraries("young_tableau" INTERFACE DDC::DDC Boost::boost sil::tensor)

add_library("sil::young_tableau" ALIAS "young_tableau")
