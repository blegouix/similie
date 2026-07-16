// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "onelab_interface.hpp"

int main(int argc, char** argv)
{
    similie::onelab_interface::OnelabInterface interface;
    return interface.run(argc, argv);
}
