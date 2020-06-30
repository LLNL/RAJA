//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

class CaptureCounterPlugin :
  public RAJA::util::Plugin2CaptureStrategy
{
  public:
  void preCapture(RAJA::util::PluginContext p) {
    if (p.platform == RAJA::Platform::host)
      std::cout << " [CaptureCounterPlugin]: Capturing host kernel for the " << ++host_capture_counter << " time!" << std::endl;
    else
      std::cout << " [CaptureCounterPlugin]: Capturing device kernel for the " << ++device_capture_counter << " time!" << std::endl;
  }

  void postCapture(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
  }

  private:
   int host_capture_counter;
   int device_capture_counter;
};

// Regiser plugin with the Plugin2CaptureRegistry
static RAJA::util::Plugin2CaptureRegistry::Add<CaptureCounterPlugin> P4Capture("capture-counter-plugin", "Capture Counter");


class LaunchCounterPlugin :
  public RAJA::util::Plugin2LaunchStrategy
{
  public:
  void preLaunch(RAJA::util::PluginContext p) {
    if (p.platform == RAJA::Platform::host)
      std::cout << " [LaunchCounterPlugin]: Launching host kernel for the " << ++host_launch_counter << " time!" << std::endl;
    else
      std::cout << " [LaunchCounterPlugin]: Launching device kernel for the " << ++device_launch_counter << " time!" << std::endl;
  }

  void postLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
  }

  private:
   int host_launch_counter;
   int device_launch_counter;
};

// Regiser plugin with the Plugin2LaunchRegistry
static RAJA::util::Plugin2LaunchRegistry::Add<LaunchCounterPlugin> P4Launch("launch-counter-plugin", "Launch Counter");
