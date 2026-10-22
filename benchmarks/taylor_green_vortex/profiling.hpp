// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later
// AI-GENERATED

#pragma once

#include <cstddef>
#include <string>

#include <Kokkos_Core.hpp>

namespace taylor_green_vortex {
inline std::string profiling_path = "TaylorGreen";

// Keep every range self-contained in nsys summaries, including nested operations.
// Host-only scopes balance push/pop even when a solver throws.
class ProfilingRegion
{
    std::size_t m_previous_size;
    bool m_active = true;

public:
    explicit ProfilingRegion(std::string const& name) : m_previous_size(profiling_path.size())
    {
        profiling_path += "/" + name;
        Kokkos::Profiling::pushRegion(profiling_path);
    }
    ProfilingRegion(ProfilingRegion const&) = delete;
    ProfilingRegion& operator=(ProfilingRegion const&) = delete;

    void end()
    {
        if (m_active) {
            Kokkos::Profiling::popRegion();
            profiling_path.resize(m_previous_size);
            m_active = false;
        }
    }
    ~ProfilingRegion()
    {
        end();
    }
};
} // namespace taylor_green_vortex
