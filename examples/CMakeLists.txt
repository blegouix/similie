# SPDX-License-Identifier: GPL-3.0

add_executable(sketch sketch.cpp)

target_link_libraries(sketch
    PUBLIC
        DDC::DDC
        sil::mesher
        sil::tensor
)
