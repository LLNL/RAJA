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

static_assert(RAJA::type_traits::is_integral<int>::value, "");
static_assert(RAJA::type_traits::is_iterator<int>::value, "");
// concepts::enable_if<
//     type_traits::is_integral<int>,
//     concepts::negate<type_traits::is_iterator<int>>>

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
      [=] (int i, int count) {
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
      [=] (int i, int count) {
        a[i] = count;
    });
  }

  ASSERT_EQ(plugin_test_capture_counter_pre, 10);
  ASSERT_EQ(plugin_test_capture_counter_post, 10);

  ASSERT_EQ(plugin_test_launch_counter_pre, 10);
  ASSERT_EQ(plugin_test_launch_counter_post, 10);

  delete[] a;
}
