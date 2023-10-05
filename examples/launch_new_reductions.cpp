//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <limits>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Reduction Example
 *
 *  This example illustrates use of the RAJA reduction types: min, max,
 *  sum, min-loc, and max-loc.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  Reduction types
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Specify the number of threads in a GPU thread block
*/
#if defined(RAJA_ENABLE_CUDA)
constexpr int CUDA_BLOCK_SIZE = 256;
#endif


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA reductions example...\n";

  // _reductions_array_init_start
//
// Define array length
//
  constexpr int N = 1000000;

//
// Allocate array data and initialize data to alternating sequence of 1, -1.
//
  int* a = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    if ( i % 2 == 0 ) {
      a[i] = 1;
    } else {
      a[i] = -1; 
    }
  }

//
// Set min and max loc values
//
  constexpr int minloc_ref = N / 2;
  a[minloc_ref] = -100;

  constexpr int maxloc_ref = N / 2 + 1;
  a[maxloc_ref] = 100;
  // _reductions_array_init_end

//
// Note: with this data initialization scheme, the following results will
//       be observed for all reduction kernels below:
//
//  - the sum will be zero
//  - the min will be -100
//  - the max will be 100
//  - the min loc will be N/2
//  - the max loc will be N/2 + 1
//
//

//
// Define index range for iterating over a elements in all examples
//
  // _reductions_range_start
  RAJA::TypedRangeSegment<int> arange(0, N);
  // _reductions_range_end

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential reductions...\n";

  // _reductions_raja_seq_start
  using EXEC_POL1   = RAJA::seq_exec;
 
  int seq_sum = 0;
  
  RAJA::forall<EXEC_POL1>
    (arange, RAJA::expt::Reduce<RAJA::operators::plus>(&seq_sum), [=](int i, int &_seq_sum) {

      _seq_sum += a[i];
      
    });

  std::cout << "\tsum = " << seq_sum << std::endl;
    // _reductions_raja_seq_end
  
//----------------------------------------------------------------------------//


  std::cout << "\n Running RAJA new reductions with launch...\n";

  //using launch_pol = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
  //using loop_pol = RAJA::LoopPolicy<RAJA::seq_exec>;

  using launch_pol = RAJA::LaunchPolicy<RAJA::omp_launch_t>;
  using loop_pol = RAJA::LoopPolicy<RAJA::omp_for_exec>;

  int launch_seq_sum = 0;
  
  RAJA::launch_params<launch_pol>
    (RAJA::LaunchParams(), RAJA::expt::Reduce<RAJA::operators::plus>(&launch_seq_sum),
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx, int &_seq_sum)
     {

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           _seq_sum += a[i]; 
         });

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           _seq_sum += 1.0; 
         });                     
     });

  std::cout << "\expexted sum N = "<< N <<" launch tsum = " << launch_seq_sum << std::endl;  
  

//
// Clean up.
//
  memoryManager::deallocate(a);

  std::cout << "\n DONE!...\n";
 
  return 0;
}
