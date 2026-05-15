// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "fake_similie_run.hpp"

namespace {

std::filesystem::path write_file(std::string const& name, std::string const& contents)
{
    std::filesystem::path const file_path = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(file_path);
    stream << contents;
    return file_path;
}

} // namespace

TEST(OnelabInterface, RectilinearMeshExportsNodePositions)
{
    std::filesystem::path const mesh_file = write_file(
            "similie_rectilinear_test.msh",
            R"($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
8
1 0 0 0
2 1 0 0
3 0 2 0
4 1 2 0
5 0 0 3
6 1 0 3
7 0 2 3
8 1 2 3
$EndNodes
$Elements
7
1 15 2 0 0 1
2 1 2 0 0 1 2
3 1 2 0 0 2 4
4 1 2 0 0 4 3
5 1 2 0 0 3 1
6 3 2 0 0 1 2 4 3
7 5 2 0 0 1 2 4 3 5 6 8 7
$EndElements
)");
    std::filesystem::path const output_file
            = std::filesystem::temp_directory_path() / "similie_rectilinear_positions_test.pos";
    std::filesystem::remove(output_file);

    similie::onelab_interface::FakeRunConfig config;
    config.input_mesh_file = mesh_file;
    config.export_view = true;
    config.output_file = output_file;

    similie::onelab_interface::FakeRunResult const result
            = similie::onelab_interface::run_fake_similie_job(config);

    EXPECT_EQ(result.status, "rectilinear mesh accepted");
    EXPECT_EQ(result.nx, 2);
    EXPECT_EQ(result.ny, 2);
    EXPECT_EQ(result.nz, 2);
    EXPECT_EQ(result.num_nodes, 8);
    EXPECT_EQ(result.num_cells, 1);
    EXPECT_DOUBLE_EQ(result.checksum, 1284.0);
    EXPECT_EQ(result.output_file, output_file);
    EXPECT_TRUE(std::filesystem::exists(output_file));

    std::ifstream stream(output_file);
    std::string
            contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    EXPECT_NE(contents.find("View \"SimiLie rectilinear node positions\""), std::string::npos);
    EXPECT_NE(contents.find("VP(1, 2, 3){1, 2, 3};"), std::string::npos);
}

TEST(OnelabInterface, NonRectilinearMeshRaisesError)
{
    std::filesystem::path const mesh_file = write_file(
            "similie_non_rectilinear_test.msh",
            R"($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
8
1 0 0 0
2 1 0 0
3 0 2 0
4 1 2 0
5 0 0 3
6 1 0 3
7 0 2 3
8 1.5 2 3
$EndNodes
$Elements
1
1 5 2 0 0 1 2 4 3 5 6 8 7
$EndElements
)");

    similie::onelab_interface::FakeRunConfig config;
    config.input_mesh_file = mesh_file;
    config.export_view = false;

    EXPECT_THROW(similie::onelab_interface::run_fake_similie_job(config), std::runtime_error);
}

TEST(OnelabInterface, NonHexahedralMeshReportsUnsupportedTopology)
{
    std::filesystem::path const mesh_file = write_file(
            "similie_tetrahedral_test.msh",
            R"($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
1
1 4 2 0 0 1 2 3 4
$EndElements
)");

    similie::onelab_interface::FakeRunConfig config;
    config.input_mesh_file = mesh_file;
    config.export_view = false;

    try {
        (void)similie::onelab_interface::run_fake_similie_job(config);
        FAIL() << "expected a runtime_error";
    } catch (std::runtime_error const& error) {
        EXPECT_NE(
                std::string(error.what()).find(
                        "requires the whole mesh to be made of quadrilaterals or hexahedra"),
                std::string::npos);
    }
}
