//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

class CounterPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preCapture(RAJA::util::PluginContext p) {
    if (p.platform == RAJA::Platform::host)
      std::cout << " [CounterPlugin]: Capturing host kernel for the " << ++host_capture_counter << " time!" << std::endl;
    else
      std::cout << " [CounterPlugin]: Capturing device kernel for the " << ++device_capture_counter << " time!" << std::endl;
  }

  void preLaunch(RAJA::util::PluginContext p) {
    if (p.platform == RAJA::Platform::host)
      std::cout << " [CounterPlugin]: Launching host kernel for the " << ++host_launch_counter << " time!" << std::endl;
    else
      std::cout << " [CounterPlugin]: Launching device kernel for the " << ++device_launch_counter << " time!" << std::endl;
  }

  private:
   int host_capture_counter;
   int device_capture_counter;
   int host_launch_counter;
   int device_launch_counter;
};

// Regiser plugin with the PluginRegistry
static RAJA::util::PluginRegistry::Add<CounterPlugin> P("counter-plugin", "Counter");

