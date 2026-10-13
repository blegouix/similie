// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <stdexcept>
#include <string>

namespace taylor_green_vortex {
enum class Physics { Incompressible, Compressible };
enum class Transport { Ideal, Resistive };
enum class Coupling { HD, EHD, MHD, FMHD };

/** Independent benchmark choices. Ideal/resistive describes electromagnetic
 * transport, not the workshop's fixed viscosity or thermal conductivity.
 * \important This documentation is fully AI-generated.
 */
struct Configuration
{
    Physics fluid = Physics::Incompressible;
    Transport transport = Transport::Ideal;
    Coupling coupling = Coupling::HD;
    bool fluid_selected = false, transport_selected = false, coupling_selected = false;

    template <class Choice>
    static void select(Choice value, Choice& selected, bool& supplied, char const* group)
    {
        if (supplied && selected != value)
            throw std::runtime_error(std::string("Conflicting ") + group + " flags");
        selected = value;
        supplied = true;
    }
    bool consume(std::string const& argument)
    {
        if (argument == "--incompressible" || argument == "--physics=incompressible")
            select(Physics::Incompressible, fluid, fluid_selected, "fluid");
        else if (argument == "--compressible" || argument == "--physics=compressible")
            select(Physics::Compressible, fluid, fluid_selected, "fluid");
        else if (argument == "--ideal")
            select(Transport::Ideal, transport, transport_selected, "transport");
        else if (argument == "--resistive")
            select(Transport::Resistive, transport, transport_selected, "transport");
        else if (argument == "--hd")
            select(Coupling::HD, coupling, coupling_selected, "coupling");
        else if (argument == "--ehd")
            select(Coupling::EHD, coupling, coupling_selected, "coupling");
        else if (argument == "--mhd")
            select(Coupling::MHD, coupling, coupling_selected, "coupling");
        else if (argument == "--fmhd")
            select(Coupling::FMHD, coupling, coupling_selected, "coupling");
        else
            return false;
        return true;
    }
    void validate() const
    {
        if (coupling == Coupling::HD && transport == Transport::Resistive)
            throw std::runtime_error("--resistive requires --ehd, --mhd, or --fmhd");
    }
    std::string name() const
    {
        std::string const sector = coupling == Coupling::HD    ? "hd"
                                   : coupling == Coupling::EHD ? "ehd"
                                   : coupling == Coupling::MHD ? "mhd"
                                                               : "fmhd";
        return std::string(fluid == Physics::Compressible ? "compressible" : "incompressible") + "_"
               + (transport == Transport::Ideal ? "ideal" : "resistive") + "_" + sector;
    }
};
} // namespace taylor_green_vortex
