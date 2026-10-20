# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

add_subdirectory(exterior)
add_subdirectory(multidomains)
add_subdirectory(physics)
add_subdirectory(solvers)
add_subdirectory(tensor)

add_library(check_concepts OBJECT concepts.cpp)
target_link_libraries(check_concepts PRIVATE sil::sil)
add_dependencies(check_concepts similie_physics_generated)
