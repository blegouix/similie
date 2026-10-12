# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

find_package(benchmark 1.7 REQUIRED CONFIG)
find_package(HDF5 REQUIRED COMPONENTS C)
add_subdirectory(taylor_green_vortex)
