// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

namespace similie::onelab_interface {

struct FakeRunConfig
{
    double tensor_scale = 1.0;
    bool export_view = true;
    std::filesystem::path output_file;
};

struct FakeRunResult
{
    std::string status;
    double checksum = 0.0;
    double max_entry = 0.0;
    std::filesystem::path output_file;
};

FakeRunResult run_fake_similie_job(FakeRunConfig const& config);

} // namespace similie::onelab_interface
