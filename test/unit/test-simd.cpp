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
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>


using namespace RAJA;
using namespace RAJA::statement;

template<typename T>
RAJA_INLINE
T alignHint(T x)
{
#if defined (RAJA_COMPILER_INTEL)
  RAJA_ALIGN_DATA(x);
  return x;
#else
  return RAJA_ALIGN_DATA(x);
#endif
}


TEST(SIMD, align){

  int N = 1024;
  double c = 0.5;
  double *a = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,N*sizeof(double));
  double *b = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,N*sizeof(double));
    
  for(int i=0; i<N; ++i) 
    {
      a[i] = 0; 
      b[i] = 2.0;
    }


  double *y = alignHint(a);
  double *x = alignHint(b);

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
      y[i] += x[i] * c;
    });

  for(int i=0; i<N; ++i)
    {
      ASSERT_FLOAT_EQ(y[i], 1.0);
    }

}
