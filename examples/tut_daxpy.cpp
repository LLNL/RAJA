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

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Daxpy Example
 *
 *  Computes a += b*c, where a, b are vectors of doubles
 *  and c is a scalar double. It illustrates similarities between a 
 *  C-style for-loop and a RAJA forall loop. 
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment 
 *    -  Execution policies
 */

//
// Functions for checking and printing results
//
void checkResult(double* v1, double* v2, int len);
void printResult(double* v, int len); 

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA daxpy example...\n";

//
// Define vector length
//
  const int N = 1000000;

//
// Allocate and initialize vector data.
//
  double* a0 = new double[N];
  double* aref = new double[N];

  double* ta = new double[N];
  double* tb = new double[N];
  
  double c = 3.14159;
  
  for (int i = 0; i < N; i++) {
    a0[i] = 1.0;
    tb[i] = 2.0;
  }

//
// Declare and set pointers to array data. 
// We reset them for each daxpy version so that 
// they all look the same.
//

  double* a = ta;
  double* b = tb;


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of daxpy...\n";
   
  std::memcpy( a, a0, N * sizeof(double) );  

  for (int i = 0; i < N; ++i) {
    a[i] += b[i] * c;
  }

  std::memcpy( aref, a, N* sizeof(double) ); 

//----------------------------------------------------------------------------//

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
  
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential daxpy...\n";
   
  std::memcpy( a, a0, N * sizeof(double) );  

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N); 


//----------------------------------------------------------------------------//

//
// RAJA SIMD version.
//
  std::cout << "\n Running RAJA SIMD daxpy...\n";
   
  std::memcpy( a, a0, N * sizeof(double) );  

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N); 


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP daxpy...\n";
   
  std::memcpy( a, a0, N * sizeof(double) );  

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N); 
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
//
// RAJA CUDA parallel GPU version (256 threads per thread block).
//
  std::cout << "\n Running RAJA CUDA daxpy...\n";

  a = 0; b = 0;
  cudaErrchk(cudaMalloc( (void**)&a, N * sizeof(double) ));
  cudaErrchk(cudaMalloc( (void**)&b, N * sizeof(double) ));
 
  cudaErrchk(cudaMemcpy( a, a0, N * sizeof(double), cudaMemcpyHostToDevice )); 
  cudaErrchk(cudaMemcpy( b, tb, N * sizeof(double), cudaMemcpyHostToDevice )); 

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) {
    a[i] += b[i] * c;
  });

  cudaErrchk(cudaMemcpy( ta, a, N * sizeof(double), cudaMemcpyDeviceToHost ));

  cudaErrchk(cudaFree(a));
  cudaErrchk(cudaFree(b));

  a = ta;
  checkResult(a, aref, N);
//printResult(a, N); 
#endif

//----------------------------------------------------------------------------//

//
// Clean up. 
//
  delete[] a0; 
  delete[] aref; 
  delete[] ta; 
  delete[] tb;
  
  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to compare result to reference and report P/F.
//
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

//
// Function to print result. 
//
void printResult(double* v, int len) 
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}
