# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later
# AI-GENERATED

add_library(multidomains INTERFACE)
target_link_libraries(multidomains INTERFACE sil::exterior)
add_library(sil::multidomains ALIAS multidomains)
