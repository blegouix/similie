# SPDX-License-Identifier: GPL-3.0

add_library("tensor" STATIC tensor.cpp)

target_include_directories("tensor" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("tensor" PUBLIC DDC::DDC)

add_library("sil::tensor" ALIAS "tensor")
