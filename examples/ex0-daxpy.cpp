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
 *  Daxpy example, a += b*c, where a, b are vectors of doubles
 *  and c is a scalar double, illustrates the similarities between a 
 *  C-style for-loop and a RAJA forall loop. 
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment 
 *    -  Execution policies
 */

void checkResult(double* v1, double* v2, int len) 
{
  bool match = true;
  for (int i = 0; i < len; i++) {
    if ( v1[i] != v2[i] ) { match = false; }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  } 
}

void printResult(double* v, int len) 
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
} 

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "RAJA example 0: daxpy\n\n";

//
// Allocate and initialize data.
//
  double* a0 = new double[1000];
  double* aref = new double[1000];

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

  std::memcpy( aref, a, 1000* sizeof(double) ); 

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

  checkResult(a, aref, 1000);
//printResult(a, 1000); 


//
// RAJA SIMD version.
//
  std::cout << "\n Running RAJA SIMD daxpy...\n";
   
  std::memcpy( a, a0, 1000 * sizeof(double) );  

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, 1000);
//printResult(a, 1000); 


#if defined(RAJA_ENABLE_OPENMP)
//
// RAJA OpenMP parallel multithreading version.
//
  std::cout << "\n Running RAJA OpenMP daxpy...\n";
   
  std::memcpy( a, a0, 1000 * sizeof(double) );  

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, 1000);
//printResult(a, 1000); 
#endif

#if defined(RAJA_ENABLE_CUDA)
//
// RAJA CUDA parallel GPU version (256 threads per thread block).
//
  std::cout << "\n Running RAJA CUDA daxpy...\n";

  a = 0; b = 0;
  cudaErrchk(cudaMalloc( (void**)&a, 1000 * sizeof(double) ));
  cudaErrchk(cudaMalloc( (void**)&b, 1000 * sizeof(double) ));
 
  cudaErrchk(cudaMemcpy( a, a0, 1000 * sizeof(double), cudaMemcpyHostToDevice )); 
  cudaErrchk(cudaMemcpy( b, tb, 1000 * sizeof(double), cudaMemcpyHostToDevice )); 

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, 1000), 
    [=] RAJA_DEVICE (int i) {
    a[i] += b[i] * c;
  });

  cudaErrchk(cudaMemcpy( ta, a, 1000 * sizeof(double), cudaMemcpyDeviceToHost ));

  cudaErrchk(cudaFree(a));
  cudaErrchk(cudaFree(b));

  a = ta;
  checkResult(a, aref, 1000);
//printResult(a, 1000); 
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

