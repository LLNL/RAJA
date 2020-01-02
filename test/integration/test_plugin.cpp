//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/RAJA.hpp"
#include "counter.hpp"

#include "gtest/gtest.h"

int plugin_test_counter_pre{0};
int plugin_test_counter_post{0};

// Check that the plugin is called the correct number of times, once before and
// after each kernel invocation
TEST(PluginTest, Counter)
{
  int* a = new int[10];

  for (int i = 0; i < 10; i++) {
    RAJA::forall<RAJA::seq_exec>(
      RAJA::RangeSegment(0,10), 
      [=] (int i) {
        a[i] = 0;
    });
  }

  ASSERT_EQ(plugin_test_counter_pre, 10);
  ASSERT_EQ(plugin_test_counter_post, 10);

  delete[] a;
}
