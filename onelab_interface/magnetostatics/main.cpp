// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include "linear_magnetostatics_onelab_interface.hpp"

int main(int argc, char** argv)
{
    similie::onelab_interface::LinearMagnetostaticsOnelabInterface interface;
    return interface.run(argc, argv);
}
