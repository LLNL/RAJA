//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic integration tests for plugins with workgroup.
///

#ifndef __TEST_PLUGIN_WORKGROUP_HPP__
#define __TEST_PLUGIN_WORKGROUP_HPP__

#include "test-plugin.hpp"


// Check that the plugin is called with the right Platform.
// Check that the plugin is called the correct number of times,
// once before and after each enqueue capture for the capture counter,
// once before and after each run invocation for the launch counter.

// test with workgroup
template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKINGRES,
          RAJA::Platform PLATFORM>
void PluginWorkGroupTestImpl()
{
  using WorkPool_type = RAJA::WorkPool<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkGroup_type = RAJA::WorkGroup<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkSite_type = RAJA::WorkSite<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  SetupPluginVars spv(WORKINGRES{});

  CounterData* data = plugin_test_resource->allocate<CounterData>(10);

  {
    CounterData loop_data[10];
    for (int i = 0; i < 10; i++) {
      loop_data[i].capture_platform_active = RAJA::Platform::undefined;
      loop_data[i].capture_counter_pre     = -1;
      loop_data[i].capture_counter_post    = -1;
      loop_data[i].launch_platform_active = RAJA::Platform::undefined;
      loop_data[i].launch_counter_pre     = -1;
      loop_data[i].launch_counter_post    = -1;
    }
    plugin_test_resource->memcpy(data, &loop_data[0], 10*sizeof(CounterData));
  }

  WorkPool_type pool(Allocator{});

  for (int i = 0; i < 10; i++) {
    pool.enqueue(RAJA::TypedRangeSegment<IndexType>{i,i+1},
        PluginTestCallable{data});
  }

  {
    CounterData plugin_data;
    plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
    ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.capture_counter_pre,     10);
    ASSERT_EQ(plugin_data.capture_counter_post,    10);
    ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.launch_counter_pre,     0);
    ASSERT_EQ(plugin_data.launch_counter_post,    0);
  }

  {
    CounterData loop_data[10];
    plugin_test_resource->memcpy(&loop_data[0], data, 10*sizeof(CounterData));

    for (int i = 0; i < 10; i++) {
      ASSERT_EQ(loop_data[i].capture_platform_active, RAJA::Platform::undefined);
      ASSERT_EQ(loop_data[i].capture_counter_pre,     -1);
      ASSERT_EQ(loop_data[i].capture_counter_post,    -1);
      ASSERT_EQ(loop_data[i].launch_platform_active, RAJA::Platform::undefined);
      ASSERT_EQ(loop_data[i].launch_counter_pre,     -1);
      ASSERT_EQ(loop_data[i].launch_counter_post,    -1);
    }
  }

  WorkGroup_type group = pool.instantiate();

  {
    CounterData plugin_data;
    plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
    ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.capture_counter_pre,     10);
    ASSERT_EQ(plugin_data.capture_counter_post,    10);
    ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.launch_counter_pre,     0);
    ASSERT_EQ(plugin_data.launch_counter_post,    0);
  }

  {
    CounterData loop_data[10];
    plugin_test_resource->memcpy(&loop_data[0], data, 10*sizeof(CounterData));

    for (int i = 0; i < 10; i++) {
      ASSERT_EQ(loop_data[i].capture_platform_active, RAJA::Platform::undefined);
      ASSERT_EQ(loop_data[i].capture_counter_pre,     -1);
      ASSERT_EQ(loop_data[i].capture_counter_post,    -1);
      ASSERT_EQ(loop_data[i].launch_platform_active, RAJA::Platform::undefined);
      ASSERT_EQ(loop_data[i].launch_counter_pre,     -1);
      ASSERT_EQ(loop_data[i].launch_counter_post,    -1);
    }
  }

  WorkSite_type site = group.run();

  {
    CounterData plugin_data;
    plugin_test_resource->memcpy(&plugin_data, plugin_test_data, sizeof(CounterData));
    ASSERT_EQ(plugin_data.capture_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.capture_counter_pre,     10);
    ASSERT_EQ(plugin_data.capture_counter_post,    10);
    ASSERT_EQ(plugin_data.launch_platform_active, RAJA::Platform::undefined);
    ASSERT_EQ(plugin_data.launch_counter_pre,     1);
    ASSERT_EQ(plugin_data.launch_counter_post,    1);
  }

  {
    CounterData loop_data[10];
    plugin_test_resource->memcpy(&loop_data, data, 10*sizeof(CounterData));

    for (int i = 0; i < 10; i++) {
      ASSERT_EQ(loop_data[i].capture_platform_active, PLATFORM);
      ASSERT_EQ(loop_data[i].capture_counter_pre,     i+1);
      ASSERT_EQ(loop_data[i].capture_counter_post,    i);
      ASSERT_EQ(loop_data[i].launch_platform_active, PLATFORM);
      ASSERT_EQ(loop_data[i].launch_counter_pre,     1);
      ASSERT_EQ(loop_data[i].launch_counter_post,    0);
    }
  }

  plugin_test_resource->deallocate(data);
}


TYPED_TEST_SUITE_P(PluginWorkGroupTest);
template <typename T>
class PluginWorkGroupTest : public ::testing::Test
{
};

TYPED_TEST_P(PluginWorkGroupTest, PluginWorkGroup)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;
  using PlatformHolder = typename camp::at<TypeParam, camp::num<6>>::type;

  PluginWorkGroupTestImpl<ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE, PlatformHolder::platform>( );
}

REGISTER_TYPED_TEST_SUITE_P(PluginWorkGroupTest,
                            PluginWorkGroup);

#endif  //__TEST_PLUGIN_WORKGROUP_HPP__
