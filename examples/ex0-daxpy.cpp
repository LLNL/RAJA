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
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Example 0: Daxpy
 *
 *  This Double-Precision-A*X Plus Y example code 
 *  illustrates the similarities between a C++ style 
 *  for loop and a RAJA forall loop. It introduces
 *  the RAJA `forall` loop iteration method and range
 *  segment concepts and RAJA execution policies.
 */

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

//
// Allocate and initialize data.
//
  double* a0 = new double[1000];

  double* ta = new double[1000];
  double* tb = new double[1000];
  
  double c = 3.14159;
  
  for (int i = 0; i < 1000; i++) {
    a0[i] = 1.0;
    tb[i] = 2.0;
  }

//
// Declare and set pointers to array data. 
// We reset them for each daxpy version so that 
// they all look the same.

  double* a = ta;
  double* b = tb;

//
// C-style daxpy operation.
//
  std::cout << "\n Running C-vesion of daxpy...\n";
   
  std::memcpy( a, a0, 1000 * sizeof(double) );  

  for (int i = 0; i < 1000; ++i) {
    a[i] += b[i] * c;
  }

//
// In the following, we show a RAJA version
// of the daxpy operation and how it can
// be run differently by choosing different
// RAJA execution policies. 
//
// Note that the only thing that changes in 
// these versions is the execution policy.
// To implement these cases using the 
// programming model choices directly, would
// require unique changes for each.
//
  
//
// RAJA version of sequential daxpy operation.
//
  std::cout << "\n Running RAJA sequential daxpy...\n";
   
  std::memcpy( a, a0, 1000 * sizeof(double) );  

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

//
// RAJA SIMD version.
//
  std::cout << "\n Running RAJA SIMD daxpy...\n";
   
  std::memcpy( a, a0, 1000 * sizeof(double) );  

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

#if defined(RAJA_ENABLE_OPENMP)
//
// RAJA OpenMP parallel multithreading version.
//
  std::cout << "\n Running RAJA OpenMP daxpy...\n";
   
  std::memcpy( a, a0, 1000 * sizeof(double) );  

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });
#endif

#if defined(RAJA_ENABLE_CUDA)
//
// RAJA CUDA parallel GPU version (256 threads per thread block).
//
  std::cout << "\n Running RAJA CUDA daxpy...\n";

  a = 0; b = 0;
  cudaMalloc( (void**)&a, 1000 * sizeof(double) ) );
  cudaMalloc( (void**)&b, 1000 * sizeof(double) ) );
 
  cudaMemcpy( a, a0, 1000 * sizeof(double), cudaMemcpyHostToDevice ); 
  cudaMemcpy( b, tb, 1000 * sizeof(double), cudaMemcpyHostToDevice ); 

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

  cudaMemcpy( ta, a, 1000 * sizeof(double), cudaMemcpyDeviceToHost );

  cudaFree(a);
  cudaFree(b);

  a = ta;
#endif

//
// Clean up. 
//
  delete[] a0; 
  delete[] ta; 
  delete[] tb;
  
  std::cout << "\n DONE!...\n";

  return 0;
}

