//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-689114
// 
// All rights reserved.
// 
// This file is part of RAJA.
// 
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

class CounterPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preLaunch(RAJA::util::PluginContext p) {
    if (p.platform == RAJA::Platform::host)
      std::cout << " [CounterPlugin]: Launching host kernel for the " << ++host_counter << " time!" << std::endl;
    else
      std::cout << " [CounterPlugin]: Launching device kernel for the " << ++device_counter << " time!" << std::endl;
  }

  void postLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
  }

  private:
   int host_counter;
   int device_counter;
};

// Regiser plugin with the PluginRegistry
static RAJA::util::PluginRegistry::Add<CounterPlugin> P("counter-plugin", "Counter");
