//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

#include "counter.hpp"

class CounterPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    plugin_test_counter_pre++;
  }

  void postLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    plugin_test_counter_post++;
  }
};

// Regiser plugin with the PluginRegistry
static RAJA::util::PluginRegistry::Add<CounterPlugin> P("counter-plugin", "Counter");
