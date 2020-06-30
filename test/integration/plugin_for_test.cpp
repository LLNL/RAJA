//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>

#include "counter.hpp"

class CaptureCounterPlugin :
  public RAJA::util::Plugin2CaptureStrategy
{
  public:
  void preCapture(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    plugin_test_capture_counter_pre++;
  }

  void postCapture(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    plugin_test_capture_counter_post++;
  }
};

// Regiser plugin with the Plugin2CaptureRegistry
static RAJA::util::Plugin2CaptureRegistry::Add<CaptureCounterPlugin> P4Capture("capture-counter-plugin", "Capture Counter");


class LaunchCounterPlugin :
  public RAJA::util::Plugin2LaunchStrategy
{
  public:
  void preLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    plugin_test_launch_counter_pre++;
  }

  void postLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    plugin_test_launch_counter_post++;
  }
};

// Regiser plugin with the Plugin2LaunchRegistry
static RAJA::util::Plugin2LaunchRegistry::Add<LaunchCounterPlugin> P4Launch("launch-counter-plugin", "Launch Counter");
