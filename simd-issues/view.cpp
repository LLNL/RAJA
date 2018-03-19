//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#include <cstdlib>
#include <iostream>
#include <chrono>

#include <time.h>       /* time */

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

using arr_type = double; 

#define nestedTest 1

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{


  srand (time(NULL));
  RAJA::Index_type stride = rand() % 50 + 1; 
  RAJA::Index_type arrLen = stride*stride*stride;  
  arr_type *A = new arr_type[arrLen];   
  arr_type *B = new arr_type[arrLen];   
  arr_type *C = new arr_type[arrLen];   
  RAJA::View<arr_type, RAJA::Layout<3> > Aview(A,stride,stride,stride);
  RAJA::View<arr_type, RAJA::Layout<3> > Bview(B,stride,stride,stride);
  RAJA::View<arr_type, RAJA::Layout<3> > Cview(C,stride,stride,stride);
  
  for(int i=0; i<arrLen; ++i){
    A[i] = 0.5;
    B[i] = 2; 
  }

  RAJA::RangeSegment myStride(0,stride);

  using POL = 
    RAJA::KernelPolicy<
    RAJA::statement::For<2, RAJA::omp_parallel_for_exec,
    RAJA::statement::For<1, RAJA::loop_exec,
    RAJA::statement::For<0, RAJA::simd_exec,
    RAJA::statement::Lambda<0> > > > >;



  RAJA::kernel<POL>(RAJA::make_tuple(myStride,myStride,myStride), [=] (RAJA::Index_type k, RAJA::Index_type j, 
                                                                       RAJA::Index_type i){
                      Cview(k,j,i) = Aview(k,j,i)*Bview(k,j,i) ;                                     
                    });

                   
  double sum = 0.0;
  for(int i=0; i<arrLen; ++i){
    sum += C[i];
  }

  std::cout<<"arrLen should equal sum"<<std::endl;
  std::cout<<"arrLen = "<<arrLen<<" "<<"sum = "<<sum<<std::endl;
  assert(std::abs(arrLen-sum) < 1e-8);

  return 0;
}
