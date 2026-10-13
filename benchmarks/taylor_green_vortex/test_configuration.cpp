// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#include <iostream>
#include <vector>

#include "configuration.hpp"

int main()
{
    using taylor_green_vortex::Configuration;
    try {
        Configuration defaults;
        defaults.validate();
        if (defaults.name() != "incompressible_ideal_hd")
            throw std::runtime_error("Incorrect default configuration");
        for (std::string const& fluid :
             std::vector<std::string> {"--incompressible", "--compressible"})
            for (std::string const& transport : std::vector<std::string> {"--ideal", "--resistive"})
                for (std::string const& coupling :
                     std::vector<std::string> {"--hd", "--ehd", "--mhd", "--fmhd"}) {
                    Configuration selected;
                    selected.consume(coupling);
                    selected.consume(transport);
                    selected.consume(fluid);
                    bool rejected = false;
                    try {
                        selected.validate();
                    } catch (std::runtime_error const&) {
                        rejected = true;
                    }
                    if (rejected != (transport == "--resistive" && coupling == "--hd"))
                        throw std::runtime_error("Incorrect configuration compatibility");
                }
        for (std::vector<std::string> const& flags : std::vector<std::vector<std::string>> {
                     {"--incompressible", "--compressible"},
                     {"--ideal", "--resistive"},
                     {"--hd", "--mhd"},
                     {"--ehd", "--fmhd"},
                     {"--mhd", "--ehd"},
                     {"--physics=compressible", "--incompressible"}}) {
            for (bool reverse : {false, true}) {
                Configuration selected;
                bool rejected = false;
                try {
                    selected.consume(flags[reverse ? 1 : 0]);
                    selected.consume(flags[reverse ? 0 : 1]);
                } catch (std::runtime_error const&) {
                    rejected = true;
                }
                if (!rejected)
                    throw std::runtime_error("Conflicting selectors accepted");
            }
        }
        if (defaults.consume("--benchmark_repetitions=2"))
            throw std::runtime_error("Google Benchmark flags must pass through");
        defaults.consume("--incompressible");
        defaults.consume("--incompressible");
        defaults.validate();
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
