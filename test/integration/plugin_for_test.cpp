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
  void preCapture(RAJA::util::PluginContext p) {
    plugin_test_capture_counter_pre++;
    plugin_test_capture_platform_active = p.platform;
  }

  void postCapture(RAJA::util::PluginContext p) {
    plugin_test_capture_counter_post++;
    ASSERT_EQ(plugin_test_capture_platform_active, p.platform);
    plugin_test_capture_platform_active = RAJA::platform::undefined;
  }

  void preLaunch(RAJA::util::PluginContext p) {
    plugin_test_launch_counter_pre++;
    plugin_test_launch_platform_active = p.platform;
  }

  void postLaunch(RAJA::util::PluginContext p) {
    plugin_test_launch_counter_post++;
    ASSERT_EQ(plugin_test_launch_platform_active, p.platform);
    plugin_test_launch_platform_active = RAJA::platform::undefined;
  }
};

// Regiser plugin with the PluginRegistry
static RAJA::util::PluginRegistry::Add<CounterPlugin> P("counter-plugin", "Counter");
