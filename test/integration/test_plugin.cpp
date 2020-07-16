//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/RAJA.hpp"
#include "counter.hpp"

#include "gtest/gtest.h"

int plugin_test_capture_counter_pre{0};
int plugin_test_capture_counter_post{0};

int plugin_test_launch_counter_pre{0};
int plugin_test_launch_counter_post{0};

RAJA::Platform plugin_test_capture_platform_active = RAJA::Platform::undefined;

RAJA::Platform plugin_test_launch_platform_active = RAJA::Platform::undefined;


// Check that the plugin is called the correct number of times,
// once before and after each kernel capture for the capture counter
// once before and after each kernel invocation for the launch counter
TEST(PluginTest, ForAllCounter)
{
  plugin_test_capture_counter_pre = 0;
  plugin_test_capture_counter_post = 0;

  plugin_test_launch_counter_pre = 0;
  plugin_test_launch_counter_post = 0;

  int* a = new int[10];

  for (int i = 0; i < 10; i++) {
    RAJA::forall<RAJA::seq_exec>(
      RAJA::RangeSegment(0,10),
      [=] (int i) {
        a[i] = 0;
    });
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 10);
  ASSERT_EQ(plugin_test_capture_counter_post, 10);

  ASSERT_EQ(plugin_test_launch_counter_pre, 10);
  ASSERT_EQ(plugin_test_launch_counter_post, 10);

  delete[] a;
}

TEST(PluginTest, ForAllICountCounter)
{
  plugin_test_capture_counter_pre = 0;
  plugin_test_capture_counter_post = 0;

  plugin_test_launch_counter_pre = 0;
  plugin_test_launch_counter_post = 0;

  int* a = new int[10];

  int icount = 0;
  for (int i = 0; i < 10; i++) {
    RAJA::forall_Icount<RAJA::seq_exec>(
      RAJA::RangeSegment(0,10), icount,
      [=] (int count, int i) {
        a[i] = count;
    });
    icount += 10;
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 10);
  ASSERT_EQ(plugin_test_capture_counter_post, 10);

  ASSERT_EQ(plugin_test_launch_counter_pre, 10);
  ASSERT_EQ(plugin_test_launch_counter_post, 10);

  delete[] a;
}

TEST(PluginTest, ForAllIdxSetCounter)
{
  plugin_test_capture_counter_pre = 0;
  plugin_test_capture_counter_post = 0;

  plugin_test_launch_counter_pre = 0;
  plugin_test_launch_counter_post = 0;

  RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

  for (int i = 0; i < 10; i++) {
    iset.push_back(RAJA::RangeSegment(0, 10));
  }

  int* a = new int[10];

  for (int i = 0; i < 10; i++) {
    RAJA::forall<RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>>(
      iset,
      [=] (int i) {
        a[i] = 0;
    });
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 10);
  ASSERT_EQ(plugin_test_capture_counter_post, 10);

  ASSERT_EQ(plugin_test_launch_counter_pre, 10);
  ASSERT_EQ(plugin_test_launch_counter_post, 10);

  delete[] a;
}

TEST(PluginTest, ForAllICountIdxSetCounter)
{
  plugin_test_capture_counter_pre = 0;
  plugin_test_capture_counter_post = 0;

  plugin_test_launch_counter_pre = 0;
  plugin_test_launch_counter_post = 0;

  RAJA::TypedIndexSet< RAJA::RangeSegment > iset;

  for (int i = 0; i < 10; i++) {
    iset.push_back(RAJA::RangeSegment(0, 10));
  }

  int* a = new int[10];

  for (int i = 0; i < 10; i++) {
    RAJA::forall_Icount<RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>>(
      iset,
      [=] (int count, int i) {
        a[i] = count;
    });
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 10);
  ASSERT_EQ(plugin_test_capture_counter_post, 10);

  ASSERT_EQ(plugin_test_launch_counter_pre, 10);
  ASSERT_EQ(plugin_test_launch_counter_post, 10);

  delete[] a;
}


TEST(PluginTest, ForAllMultiPolicyCounter)
{
  plugin_test_capture_counter_pre = 0;
  plugin_test_capture_counter_post = 0;

  plugin_test_launch_counter_pre = 0;
  plugin_test_launch_counter_post = 0;

  int* a = new int[10];

  auto mp = RAJA::make_multi_policy<RAJA::seq_exec, RAJA::loop_exec>(
      [](const RAJA::RangeSegment &r) {
        if (r.size() < 10) {
          return 0;
        } else {
          return 1;
        }
      });

  for (int i = 0; i < 10; i++) {
    RAJA::forall(
      mp, RAJA::RangeSegment(0, 9),
      [=] (int i) {
        a[i] = 0;
    });
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 10);
  ASSERT_EQ(plugin_test_capture_counter_post, 10);

  ASSERT_EQ(plugin_test_launch_counter_pre, 10);
  ASSERT_EQ(plugin_test_launch_counter_post, 10);


  for (int i = 0; i < 10; i++) {
    RAJA::forall(
      mp, RAJA::RangeSegment(0, 10),
      [=] (int i) {
        a[i] = 0;
    });
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 20);
  ASSERT_EQ(plugin_test_capture_counter_post, 20);

  ASSERT_EQ(plugin_test_launch_counter_pre, 20);
  ASSERT_EQ(plugin_test_launch_counter_post, 20);

  delete[] a;
}
