# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0

add_executable(sketch sketch.cpp)

target_link_libraries(sketch
    PUBLIC
        DDC::DDC
        sil::mesher
        sil::tensor
)

add_executable(riemann_contraction riemann_contraction.cpp)

target_link_libraries(riemann_contraction
    PUBLIC
        DDC::DDC
        sil::tensor
)
