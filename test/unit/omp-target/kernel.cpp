//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;
using namespace RAJA::statement;

TEST(Kernel, omptarget)
{

  using Pol = RAJA::KernelPolicy<
                For<0, RAJA::omp_target_parallel_for_exec<64> >,
                For<1, RAJA::loop_exec>
              >;

  double* array = new double[25*25];

#pragma omp target enter data map(to: array[0:25*25])
#pragma omp target data use_device_ptr(array)

  RAJA::kernel<Pol>(
      RAJA::make_tuple(
        RAJA::RangeSegment(0,25),
        RAJA::RangeSegment(0,25)),
      [=] (int i, int j) {
      //array[i + (25*j)] = i*j;
      int idx = i*j;

      //array[0] = i*j;
  });


//#pragma omp target update from(array[:25*25])
//  for (int i = 0; i < 25*25; i++) {
//    std::cout << i << "=" << array[i] << std::endl;
//  }
}

