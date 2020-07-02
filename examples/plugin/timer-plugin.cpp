//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>
#include <chrono>

class TimerPlugin : public RAJA::util::PluginStrategy
{
public:
  void preLaunch(RAJA::util::PluginContext& RAJA_UNUSED_ARG(p))
  {
    start_time = std::chrono::steady_clock::now();
  }

  void postLaunch(RAJA::util::PluginContext& p)
  {
    end_time = std::chrono::steady_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (p.platform == RAJA::Platform::host)
      printf("[TimerPlugin]: Elapsed time of host kernel was %f ms\n", elapsedMs);
    else
      printf("[TimerPlugin]: Elapsed time of device kernel was %f ms\n", elapsedMs);
  }

private:
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;
};

// Factory function for RuntimePluginLoader.
extern "C" RAJA::util::PluginStrategy *getPlugin()
{
  return new TimerPlugin;
}