//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>
#include <chrono>

class TimerPlugin : public RAJA::util::PluginStrategy
{
public:
  void preLaunch(const RAJA::util::PluginContext& RAJA_UNUSED_ARG(p)) override
  {
    start_time = std::chrono::steady_clock::now();
  }

  void postLaunch(const RAJA::util::PluginContext& p) override
  {
    end_time = std::chrono::steady_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (p.platform == RAJA::Platform::host)
    {
      printf("[TimerPlugin]: Elapsed time of host kernel was %f ms\n", elapsedMs);
    }
    else
    {
      printf("[TimerPlugin]: Elapsed time of device kernel was %f ms\n", elapsedMs);
    }
  }

private:
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
};

// Dynamically loading plugin.
extern "C" RAJA::util::PluginStrategy *getPlugin()
{
  return new TimerPlugin;
}

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<TimerPlugin> P("Timer", "Prints elapsed time of kernel executions.");