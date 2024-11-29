# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

add_subdirectory(csr)
add_subdirectory(exterior)
add_subdirectory(mesher)
add_subdirectory(tensor)
if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
add_subdirectory(young_tableau)
endif()
