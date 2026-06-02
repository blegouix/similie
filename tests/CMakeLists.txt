# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

add_subdirectory(csr)
add_subdirectory(exterior)
add_subdirectory(mesher)
add_subdirectory(tensor)
if("${SIMILIE_BUILD_YOUNG_TABLEAU}")
add_subdirectory(young_tableau)
endif()
