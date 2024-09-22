# SPDX-License-Identifier: GPL-3.0

add_library("mesher" STATIC mesher.cpp)

target_include_directories("mesher" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries("mesher" PUBLIC DDC::DDC)

add_library("sil::mesher" ALIAS "mesher")
