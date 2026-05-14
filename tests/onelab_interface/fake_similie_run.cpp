// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "fake_similie_run.hpp"

TEST(OnelabInterface, FakeRunExportsExpectedView)
{
    std::filesystem::path const output_file
            = std::filesystem::temp_directory_path() / "similie_fake_result_test.pos";
    std::filesystem::remove(output_file);

    similie::onelab_interface::FakeRunConfig config;
    config.tensor_scale = 2.0;
    config.export_view = true;
    config.output_file = output_file;

    similie::onelab_interface::FakeRunResult const result
            = similie::onelab_interface::run_fake_similie_job(config);

    EXPECT_EQ(result.status, "fake tensor allocation completed");
    EXPECT_DOUBLE_EQ(result.max_entry, 288.0);
    EXPECT_DOUBLE_EQ(result.checksum, 7834.5);
    EXPECT_EQ(result.output_file, output_file);
    EXPECT_TRUE(std::filesystem::exists(output_file));

    std::ifstream stream(output_file);
    std::string
            contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    EXPECT_NE(contents.find("View \"SimiLie fake result\""), std::string::npos);
    EXPECT_NE(contents.find("7834.5"), std::string::npos);
}
