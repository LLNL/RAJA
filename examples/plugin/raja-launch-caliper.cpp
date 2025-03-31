//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <caliper/cali.h>

#include "RAJA/RAJA.hpp"

/*
 *  Daxpy example with Caliper annotations.
 *
 *  This example repeats the RAJA daxpy example
 *  using RAJA launch and with Caliper annotations.
 *  For a sample run: CALI_CONFIG=runtime-report ./bin/raja-launch-caliper
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
// Define compute grid
//
  const int Nthreads = 256;
  const int Nteams   = (N-1)/Nthreads + 1;

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

  CALI_MARK_BEGIN("C-version elapsed time");    
  for (int i = 0; i < N; ++i) {
    a[i] += b[i] * c;
  }
  CALI_MARK_END("C-version elapsed time");
  
  std::memcpy( aref, a, N* sizeof(double) );

//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//

  std::cout << "\n Running launch sequential daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );
  using seq_launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
  using seq_loop_policy   = RAJA::LoopPolicy<RAJA::seq_exec>;

  RAJA::launch<seq_launch_policy>
    (RAJA::LaunchParams(RAJA::Teams(Nteams), RAJA::Threads(Nthreads)),
     RAJA::expt::Name("RAJA launch Seq daxpy Kernel"),
     [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

       RAJA::loop<seq_loop_policy>(ctx, RAJA::RangeSegment(0, N), [&] (int i)
       {
         a[i] += b[i] * c;
       });

     });

  checkResult(a, aref, N);
  //printResult(a, N);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
//
// RAJA CUDA parallel GPU version (256 threads per thread block).
//
  std::cout << "\n Running RAJA CUDA daxpy...\n";
  const bool async = false;
  using cuda_launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async>>;
  using cuda_loop_policy   = RAJA::LoopPolicy<RAJA::cuda_global_thread_x>;

  a = 0; b = 0;
  cudaErrchk(cudaMalloc( (void**)&a, N * sizeof(double) ));
  cudaErrchk(cudaMalloc( (void**)&b, N * sizeof(double) ));

  cudaErrchk(cudaMemcpy( a, a0, N * sizeof(double), cudaMemcpyHostToDevice ));
  cudaErrchk(cudaMemcpy( b, tb, N * sizeof(double), cudaMemcpyHostToDevice ));

   RAJA::launch<cuda_launch_policy>
     (RAJA::LaunchParams(RAJA::Teams(Nteams), RAJA::Threads(Nthreads)),
      RAJA::expt::Name("RAJA launch CUDA daxpy Kernel"),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<cuda_loop_policy>(ctx, RAJA::RangeSegment(0, N), [&] (int i)
        {
          a[i] += b[i] * c;
        });

    });

  cudaErrchk(cudaMemcpy( ta, a, N * sizeof(double), cudaMemcpyDeviceToHost ));

  cudaErrchk(cudaFree(a));
  cudaErrchk(cudaFree(b));

  a = ta;
  checkResult(a, aref, N);
//printResult(a, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
//
// RAJA HIP parallel GPU version (256 threads per thread block).
//
  std::cout << "\n Running RAJA HIP daxpy...\n";
  const bool async = false;
  using hip_launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async>>;
  using hip_loop_policy   = RAJA::LoopPolicy<RAJA::hip_global_thread_x>;

  a = 0; b = 0;
  hipErrchk(hipMalloc( (void**)&a, N * sizeof(double) ));
  hipErrchk(hipMalloc( (void**)&b, N * sizeof(double) ));

  hipErrchk(hipMemcpy( a, a0, N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( b, tb, N * sizeof(double), hipMemcpyHostToDevice ));

   RAJA::launch<hip_launch_policy>
     (RAJA::LaunchParams(RAJA::Teams(Nteams), RAJA::Threads(Nthreads)),
      RAJA::expt::Name("RAJA launch HIP daxpy Kernel"),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

        RAJA::loop<hip_loop_policy>(ctx, RAJA::RangeSegment(0, N), [&] (int i)
        {
          a[i] += b[i] * c;
        });

    });

  hipErrchk(hipMemcpy( ta, a, N * sizeof(double), hipMemcpyDeviceToHost ));

  hipErrchk(hipFree(a));
  hipErrchk(hipFree(b));

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
