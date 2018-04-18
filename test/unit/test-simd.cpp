//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#include <cstdio>
#include <time.h>       /* time */
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>


using namespace RAJA;
using namespace RAJA::statement;

#if defined(RAJA_ENABLE_OPENMP)


TEST(SIMD, saxpy){

  int N = 1024;
  double c = 0.5;
  double *a = new double[N];
  double *b = new double[N];
  for(int i=0; i<N; ++i) 
    {
      a[i] = 0; 
      b[i] = 2.0;
    }

  RAJA::forall<RAJA::omp_parallel_for_simd_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
      a[i] += b[i] * c;
    });


  for(int i=0; i<N; ++i)
    {
      ASSERT_FLOAT_EQ(a[i], 1.0);
    }

}

TEST(SIMD, copy){

  int N = 1024*2;
  double *a = new double[N];
  double *b = new double[N];
  for(int i=0; i<N; ++i) 
    {
      a[i] = 0; 
      b[i] = 10.0;
    }

  RAJA::forall<RAJA::omp_parallel_for_simd_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
      a[i] = b[i];
    });


  for(int i=0; i<N; ++i)
    {
      ASSERT_FLOAT_EQ(a[i], 10.0);
    }

}



#endif
