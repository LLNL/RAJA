//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/RAJA.hpp"


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv)) 
{
  constexpr std::size_t N{1024};

  double* a = new double[N];
  double* b = new double[N];
  double c = 3.14159;
  
  for (std::size_t i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  RAJA::forall<RAJA::loop_exec>(
    RAJA::RangeSegment(0, N),
    [=] RAJA_HOST_DEVICE (std::size_t i) {
      a[i] += b[i] * c;
    }
  );

  delete[] a;
  delete[] b;
}
