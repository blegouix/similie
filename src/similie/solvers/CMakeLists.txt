# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(solvers INTERFACE)
target_link_libraries(solvers INTERFACE MPI::MPI_CXX)
