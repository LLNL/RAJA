//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

TEST(PluginTestKokkos, Exception)
{
  int* a = new int[10];

  ASSERT_ANY_THROW({
    RAJA::forall<RAJA::seq_exec>(
        RAJA::RangeSegment(0, 10), [=](int i) { a[i] = 0; });
  });

  delete[] a;
}
