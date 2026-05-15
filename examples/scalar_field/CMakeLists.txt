# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_executable(2d_scalar_field 2d_scalar_field.cpp)
add_dependencies(2d_scalar_field similie_physics_generated)

target_link_libraries(2d_scalar_field
    PUBLIC
        DDC::core
        DDC::pdi
        DDC::splines
        sil::sil
)
