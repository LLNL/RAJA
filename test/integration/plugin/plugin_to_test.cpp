//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
  void preCapture(const RAJA::util::PluginContext& p, const RAJA::resources::Resource&) override {
    ASSERT_NE(plugin_test_data, nullptr);
    ASSERT_NE(plugin_test_resource, nullptr);

    CounterData data;
    plugin_test_resource->memcpy(&data, plugin_test_data, sizeof(CounterData));

    ASSERT_EQ(data.capture_platform_active, RAJA::Platform::undefined);
    data.capture_counter_pre++;
    data.capture_platform_active = p.platform;

    plugin_test_resource->memcpy(plugin_test_data, &data, sizeof(CounterData));
  }

  void postCapture(const RAJA::util::PluginContext& p, const RAJA::resources::Resource&) override {
    ASSERT_NE(plugin_test_data, nullptr);
    ASSERT_NE(plugin_test_resource, nullptr);

    CounterData data;
    plugin_test_resource->memcpy(&data, plugin_test_data, sizeof(CounterData));

    ASSERT_EQ(data.capture_platform_active, p.platform);
    data.capture_counter_post++;
    data.capture_platform_active = RAJA::Platform::undefined;

    plugin_test_resource->memcpy(plugin_test_data, &data, sizeof(CounterData));
  }

  void preLaunch(const RAJA::util::PluginContext& p, const RAJA::resources::Resource&) override {
    ASSERT_NE(plugin_test_data, nullptr);
    ASSERT_NE(plugin_test_resource, nullptr);

    CounterData data;
    plugin_test_resource->memcpy(&data, plugin_test_data, sizeof(CounterData));

    ASSERT_EQ(data.launch_platform_active, RAJA::Platform::undefined);
    data.launch_counter_pre++;
    data.launch_platform_active = p.platform;

    plugin_test_resource->memcpy(plugin_test_data, &data, sizeof(CounterData));
  }

  void postLaunch(const RAJA::util::PluginContext& p, const RAJA::resources::Resource&) override {
    ASSERT_NE(plugin_test_data, nullptr);
    ASSERT_NE(plugin_test_resource, nullptr);

    CounterData data;
    plugin_test_resource->memcpy(&data, plugin_test_data, sizeof(CounterData));

    ASSERT_EQ(data.launch_platform_active, p.platform);
    data.launch_counter_post++;
    data.launch_platform_active = RAJA::Platform::undefined;

    plugin_test_resource->memcpy(plugin_test_data, &data, sizeof(CounterData));
  }
};

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<CounterPlugin> P("counter-plugin", "Counter");
