// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace similie::onelab_interface {

struct FakeRunConfig
{
    std::filesystem::path input_mesh_file;
    bool export_view = true;
    std::filesystem::path output_file;
};

struct FakeRunResult
{
    std::string status;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    std::size_t num_nodes = 0;
    std::size_t num_cells = 0;
    double checksum = 0.0;
    std::filesystem::path output_file;
};

FakeRunResult run_fake_similie_job(FakeRunConfig const& config);

} // namespace similie::onelab_interface
