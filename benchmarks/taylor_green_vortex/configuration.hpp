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
    bool staged_laplacian = false, staged_discrete_hodge_star = false;
    bool staged_reduction = false, staged_reconstruction = false, staged_hodge_star = false;
    bool staged_diagonal_reduction = false, staged_diagonal_reconstruction = false;

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
        else if (argument == "--staged-laplacian" || argument == "--staged")
            staged_laplacian = true;
        else if (argument == "--staged-discrete-hodge-star")
            staged_discrete_hodge_star = true;
        else if (argument == "--staged-reduction")
            staged_reduction = true;
        else if (argument == "--staged-reconstruction")
            staged_reconstruction = true;
        else if (argument == "--staged-hodge-star")
            staged_hodge_star = true;
        else if (argument == "--staged-diagonal-reduction")
            staged_reduction = staged_diagonal_reduction = true;
        else if (argument == "--staged-diagonal-reconstruction")
            staged_reconstruction = staged_diagonal_reconstruction = true;
        else
            return false;
        return true;
    }
    int reduction_stage() const
    {
        return staged_diagonal_reduction ? 2 : int(staged_reduction);
    }
    int reconstruction_stage() const
    {
        return staged_diagonal_reconstruction ? 2 : int(staged_reconstruction);
    }
    int laplacian_mode() const
    {
        if (staged_laplacian)
            return 19;
        if (staged_discrete_hodge_star)
            return 18;
        return reduction_stage() + 3 * reconstruction_stage() + 9 * int(staged_hodge_star);
    }
    std::string staging_name() const
    {
        std::string result;
        auto append = [&](bool selected, char const* name) {
            if (selected) {
                if (!result.empty())
                    result += "+";
                result += name;
            }
        };
        append(staged_laplacian, "laplacian");
        append(staged_discrete_hodge_star, "discrete-hodge-star");
        append(staged_reduction && !staged_diagonal_reduction, "reduction");
        append(staged_diagonal_reduction, "diagonal-reduction");
        append(staged_reconstruction && !staged_diagonal_reconstruction, "reconstruction");
        append(staged_diagonal_reconstruction, "diagonal-reconstruction");
        append(staged_hodge_star, "hodge-star");
        return result.empty() ? "none" : result;
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
