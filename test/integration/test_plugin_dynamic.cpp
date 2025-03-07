//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

TEST(PluginTestDynamic, Exception)
{
  RAJA::util::init_plugins("../../lib/libdynamic_plugin.so");
  int* a = new int[10];

  ASSERT_ANY_THROW({
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10),
                               [=](int i) { a[i] = 0; });
  });

  delete[] a;
}
