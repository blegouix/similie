// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include "onelab_interface.hpp"

int main(int argc, char** argv)
{
    similie::onelab_interface::OnelabInterface interface;
    return interface.run(argc, argv);
}
