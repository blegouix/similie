// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/tensor/tensor.hpp>

#include "fake_similie_run.hpp"

namespace similie::onelab_interface {
namespace {

struct TimeTag
{
};

struct XTag
{
};

struct YTag
{
};

struct ZTag
{
};

struct Alpha : sil::tensor::TensorNaturalIndex<XTag, YTag, ZTag>
{
};

struct Mu : sil::tensor::TensorNaturalIndex<TimeTag, XTag, YTag, ZTag>
{
};

std::filesystem::path default_output_file()
{
    return std::filesystem::path(SIMILIE_ONELAB_DEFAULT_OUTPUT_DIR) / "similie_fake_result.pos";
}

void write_garbage_view(std::filesystem::path const& output_file, FakeRunResult const& result)
{
    if (!output_file.parent_path().empty()) {
        std::filesystem::create_directories(output_file.parent_path());
    }

    std::ofstream stream(output_file);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output view file: " + output_file.string());
    }

    stream << "View \"SimiLie fake result\" {\n";
    stream << "  SP(0, 0, 0){" << result.checksum << "};\n";
    stream << "  SP(1, 0, 0){" << result.max_entry << "};\n";
    stream << "  SL(0, 0, 0, 1, 0, 0){" << result.checksum << ", " << result.max_entry << "};\n";
    stream << "};\n";
}

} // namespace

FakeRunResult run_fake_similie_job(FakeRunConfig const& config)
{
    sil::tensor::TensorAccessor<Alpha, Mu> tensor_accessor;
    ddc::DiscreteDomain<Alpha, Mu> const tensor_domain = tensor_accessor.domain();
    ddc::Chunk tensor_alloc(tensor_domain, ddc::HostAllocator<double>());
    sil::tensor::Tensor tensor(tensor_alloc);

    std::array<double, 12> const seed_values = {
            0.25,
            1.00,
            2.00,
            3.50,
            5.00,
            8.00,
            13.00,
            21.00,
            34.00,
            55.00,
            89.00,
            144.00,
    };

    std::vector<double> populated_values;
    populated_values.reserve(seed_values.size());

    auto store = [&](auto first, auto second, double seed_value) {
        double const value = config.tensor_scale * seed_value;
        tensor(tensor.access_element<decltype(first), decltype(second)>()) = value;
        populated_values.push_back(value);
    };

    store(XTag(), TimeTag(), seed_values[0]);
    store(XTag(), XTag(), seed_values[1]);
    store(XTag(), YTag(), seed_values[2]);
    store(XTag(), ZTag(), seed_values[3]);
    store(YTag(), TimeTag(), seed_values[4]);
    store(YTag(), XTag(), seed_values[5]);
    store(YTag(), YTag(), seed_values[6]);
    store(YTag(), ZTag(), seed_values[7]);
    store(ZTag(), TimeTag(), seed_values[8]);
    store(ZTag(), XTag(), seed_values[9]);
    store(ZTag(), YTag(), seed_values[10]);
    store(ZTag(), ZTag(), seed_values[11]);

    double checksum = 0.0;
    for (std::size_t i = 0; i < populated_values.size(); ++i) {
        checksum += populated_values[i] * static_cast<double>(i + 1);
    }

    FakeRunResult result;
    result.status = "fake tensor allocation completed";
    result.checksum = checksum;
    result.max_entry = *std::max_element(populated_values.begin(), populated_values.end());
    result.output_file = config.output_file.empty() ? default_output_file() : config.output_file;

    if (config.export_view) {
        write_garbage_view(result.output_file, result);
    }

    return result;
}

} // namespace similie::onelab_interface
