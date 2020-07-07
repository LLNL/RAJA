//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// _plugin_example_start
#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

class CounterPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preLaunch(RAJA::util::PluginContext& p) {
    if (p.platform == RAJA::Platform::host)
      std::cout << " [CounterPlugin]: Launching host kernel for the " << ++host_counter << " time!" << std::endl;
    else
      std::cout << " [CounterPlugin]: Launching device kernel for the " << ++device_counter << " time!" << std::endl;
  }

  void postLaunch(RAJA::util::PluginContext& RAJA_UNUSED_ARG(p)) {
  }

  private:
   int host_counter;
   int device_counter;
};

// Regiser plugin with the PluginRegistry
static RAJA::util::PluginRegistry::add<CounterPlugin> P("counter-plugin", "Counter");
// _plugin_example_end
