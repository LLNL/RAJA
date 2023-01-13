//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include <cstdlib>

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  RAJA::util::init_plugins("../lib/libtimer_plugin.so");

  double *a = new double[10];
  for (int i = 0; i < 4; i++)
  {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
      a[i] = 0;
    });
  }
}
