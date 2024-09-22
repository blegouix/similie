# SPDX-License-Identifier: GPL-3.0

add_library("tensor" INTERFACE)

target_include_directories("tensor" INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("tensor" INTERFACE DDC::DDC)

add_library("sil::tensor" ALIAS "tensor")
