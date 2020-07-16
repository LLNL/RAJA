//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/util/PluginStrategy.hpp"

#include "gtest/gtest.h"

#include <iostream>

#include "counter.hpp"

class CounterPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preCapture(RAJA::util::PluginContext p) {
    ASSERT_NE(plugin_test_capture_counter_pre, nullptr);
    ASSERT_NE(plugin_test_capture_platform_active, nullptr);
    ASSERT_EQ(*plugin_test_capture_platform_active, RAJA::Platform::undefined);
    (*plugin_test_capture_counter_pre)++;
    (*plugin_test_capture_platform_active) = p.platform;
  }

  void postCapture(RAJA::util::PluginContext p) {
    ASSERT_NE(plugin_test_capture_counter_post, nullptr);
    ASSERT_NE(plugin_test_capture_platform_active, nullptr);
    ASSERT_EQ(*plugin_test_capture_platform_active, p.platform);
    (*plugin_test_capture_counter_post)++;
    (*plugin_test_capture_platform_active) = RAJA::Platform::undefined;
  }

  void preLaunch(RAJA::util::PluginContext p) {
    ASSERT_NE(plugin_test_launch_counter_pre, nullptr);
    ASSERT_NE(plugin_test_launch_platform_active, nullptr);
    ASSERT_EQ(*plugin_test_launch_platform_active, RAJA::Platform::undefined);
    (*plugin_test_launch_counter_pre)++;
    (*plugin_test_launch_platform_active) = p.platform;
  }

  void postLaunch(RAJA::util::PluginContext p) {
    ASSERT_NE(plugin_test_launch_counter_post, nullptr);
    ASSERT_NE(plugin_test_launch_platform_active, nullptr);
    ASSERT_EQ((*plugin_test_launch_platform_active), p.platform);
    (*plugin_test_launch_counter_post)++;
    (*plugin_test_launch_platform_active) = RAJA::Platform::undefined;
  }
};

// Regiser plugin with the PluginRegistry
static RAJA::util::PluginRegistry::Add<CounterPlugin> P("counter-plugin", "Counter");
