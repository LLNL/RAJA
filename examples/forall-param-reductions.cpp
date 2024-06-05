//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
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

#if defined(RAJA_ENABLE_HIP)
constexpr int HIP_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_SYCL)
constexpr int SYCL_BLOCK_SIZE = 256;
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
  RAJA::resources::Host host_res;
  int* a = host_res.allocate<int>(N);

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

//
// Define ValLoc Type
//

  using VALLOC_INT = RAJA::expt::ValLoc<int>;
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential reductions...\n";

  // _reductions_raja_seq_start
  using EXEC_POL1   = RAJA::seq_exec;

  int seq_sum = 0;
  int seq_min = std::numeric_limits<int>::max();
  int seq_max = std::numeric_limits<int>::min();
  VALLOC_INT seq_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT seq_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL1>(arange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&seq_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_maxloc),
    RAJA::expt::KernelName("RAJA Reduce Seq Kernel"),
    [=](int i, int &_seq_sum, int &_seq_min, int &_seq_max, VALLOC_INT &_seq_minloc, VALLOC_INT &_seq_maxloc) {
      _seq_sum += a[i];

      _seq_min = RAJA_MIN(a[i], _seq_min);
      _seq_max = RAJA_MAX(a[i], _seq_max);

      _seq_minloc = RAJA_MIN(VALLOC_INT(a[i], i), _seq_minloc);
      _seq_maxloc = RAJA_MAX(VALLOC_INT(a[i], i), _seq_maxloc);
      //_seq_minloc.min(a[i], i);
      //_seq_maxloc.max(a[i], i);
      // Note : RAJA::expt::ValLoc<T> objects provide min() and max() methods
      //        that are equivalent to the assignments with RAJA_MIN and RAJA_MAX
      //        above.
    }
  );

  std::cout << "\tsum = " << seq_sum << std::endl;
  std::cout << "\tmin = " << seq_min << std::endl;
  std::cout << "\tmax = " << seq_max << std::endl;
  std::cout << "\tmin, loc = " << seq_minloc.getVal() << " , "
                               << seq_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << seq_maxloc.getVal() << " , "
                               << seq_maxloc.getLoc() << std::endl;
  // _reductions_raja_seq_end


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP reductions...\n";

  // _reductions_raja_omppolicy_start
  using EXEC_POL2   = RAJA::omp_parallel_for_exec;
  // _reductions_raja_omppolicy_end

  int omp_sum = 0;
  int omp_min = std::numeric_limits<int>::max();
  int omp_max = std::numeric_limits<int>::min();
  VALLOC_INT omp_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT omp_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL2>(arange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&omp_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&omp_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&omp_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&omp_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&omp_maxloc),
    RAJA::expt::KernelName("RAJA Reduce OpenMP Kernel"),
    [=](int i, int &_omp_sum, int &_omp_min, int &_omp_max, VALLOC_INT &_omp_minloc, VALLOC_INT &_omp_maxloc) {
      _omp_sum += a[i];

      _omp_min = RAJA_MIN(a[i], _omp_min);
      _omp_max = RAJA_MAX(a[i], _omp_max);

      _omp_minloc = RAJA_MIN(VALLOC_INT(a[i], i), _omp_minloc);
      _omp_maxloc = RAJA_MAX(VALLOC_INT(a[i], i), _omp_maxloc);
      //_omp_minloc.min(a[i], i);
      //_omp_maxloc.max(a[i], i);
    }
  );

  std::cout << "\tsum = " << omp_sum << std::endl;
  std::cout << "\tmin = " << omp_min << std::endl;
  std::cout << "\tmax = " << omp_max << std::endl;
  std::cout << "\tmin, loc = " << omp_minloc.getVal() << " , "
                               << omp_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << omp_maxloc.getVal() << " , "
                               << omp_maxloc.getLoc() << std::endl;

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  std::cout << "\n Running RAJA OpenMP Target reductions...\n";

  // _reductions_raja_omppolicy_start
  using EXEC_POL3   = RAJA::omp_target_parallel_for_exec_nt;
  // _reductions_raja_omppolicy_end

  int omp_t_sum = 0;
  int omp_t_min = std::numeric_limits<int>::max();
  int omp_t_max = std::numeric_limits<int>::min();
  VALLOC_INT omp_t_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT omp_t_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL3>(arange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&omp_t_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&omp_t_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&omp_t_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&omp_t_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&omp_t_maxloc),
    [=](int i, int &_omp_t_sum, int &_omp_t_min, int &_omp_t_max, VALLOC_INT &_omp_t_minloc, VALLOC_INT &_omp_t_maxloc) {
      _omp_t_sum += a[i];

      _omp_t_min = RAJA_MIN(a[i], _omp_t_min);
      _omp_t_max = RAJA_MAX(a[i], _omp_t_max);

      _omp_t_minloc = RAJA_MIN(VALLOC_INT(a[i], i), _omp_t_minloc);
      _omp_t_maxloc = RAJA_MAX(VALLOC_INT(a[i], i), _omp_t_maxloc);
      //_omp_t_minloc.min(a[i], i);
      //_omp_t_maxloc.max(a[i], i);
    }
  );

  std::cout << "\tsum = " << omp_t_sum << std::endl;
  std::cout << "\tmin = " << omp_t_min << std::endl;
  std::cout << "\tmax = " << omp_t_max << std::endl;
  std::cout << "\tmin, loc = " << omp_t_minloc.getVal() << " , "
                               << omp_t_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << omp_t_maxloc.getVal() << " , "
                               << omp_t_maxloc.getLoc() << std::endl;

#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA reductions...\n";

  // _reductions_raja_cudapolicy_start
  using EXEC_POL3   = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  // _reductions_raja_cudapolicy_end

  int cuda_sum = 0;
  int cuda_min = std::numeric_limits<int>::max();
  int cuda_max = std::numeric_limits<int>::min();
  VALLOC_INT cuda_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT cuda_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL3>(arange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&cuda_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_maxloc),
    RAJA::expt::KernelName("RAJA Reduce CUDA Kernel"),
    [=] RAJA_DEVICE (int i, int &_cuda_sum, int &_cuda_min, int &_cuda_max, VALLOC_INT &_cuda_minloc, VALLOC_INT &_cuda_maxloc) {
      _cuda_sum += a[i];

      _cuda_min = RAJA_MIN(a[i], _cuda_min);
      _cuda_max = RAJA_MAX(a[i], _cuda_max);

      _cuda_minloc = RAJA_MIN(VALLOC_INT(a[i], i), _cuda_minloc);
      _cuda_maxloc = RAJA_MAX(VALLOC_INT(a[i], i), _cuda_maxloc);
      //_cuda_minloc.min(a[i], i);
      //_cuda_maxloc.max(a[i], i);
    }
  );

  std::cout << "\tsum = " << cuda_sum << std::endl;
  std::cout << "\tmin = " << cuda_min << std::endl;
  std::cout << "\tmax = " << cuda_max << std::endl;
  std::cout << "\tmin, loc = " << cuda_minloc.getVal() << " , "
                               << cuda_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << cuda_maxloc.getVal() << " , "
                               << cuda_maxloc.getLoc() << std::endl;

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP reductions...\n";

  int* d_a = memoryManager::allocate_gpu<int>(N);
  hipErrchk(hipMemcpy( d_a, a, N * sizeof(int), hipMemcpyHostToDevice ));

  // _reductions_raja_hippolicy_start
  using EXEC_POL3   = RAJA::hip_exec<HIP_BLOCK_SIZE>;
  // _reductions_raja_hippolicy_end

  int hip_sum = 0;
  int hip_min = std::numeric_limits<int>::max();
  int hip_max = std::numeric_limits<int>::min();
  VALLOC_INT hip_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT hip_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL3>(arange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&hip_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&hip_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&hip_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&hip_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&hip_maxloc),
    RAJA::expt::KernelName("RAJA Reduce HIP Kernel"),
    [=] RAJA_DEVICE (int i, int &_hip_sum, int &_hip_min, int &_hip_max, VALLOC_INT &_hip_minloc, VALLOC_INT &_hip_maxloc) {
      _hip_sum += d_a[i];

      _hip_min = RAJA_MIN(d_a[i], _hip_min);
      _hip_max = RAJA_MAX(d_a[i], _hip_max);

      _hip_minloc = RAJA_MIN(VALLOC_INT(d_a[i], i), _hip_minloc);
      _hip_maxloc = RAJA_MAX(VALLOC_INT(d_a[i], i), _hip_maxloc);
      //_hip_minloc.min(d_a[i], i);
      //_hip_maxloc.max(d_a[i], i);
    }
  );

  std::cout << "\tsum = " << hip_sum << std::endl;
  std::cout << "\tmin = " << hip_min << std::endl;
  std::cout << "\tmax = " << hip_max << std::endl;
  std::cout << "\tmin, loc = " << hip_minloc.getVal() << " , "
                               << hip_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << hip_maxloc.getVal() << " , "
                               << hip_maxloc.getLoc() << std::endl;

  memoryManager::deallocate_gpu(d_a);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_SYCL)
  std::cout << "\n Running RAJA SYCL reductions...\n";

  RAJA::resources::Sycl sycl_res;

  int* d_a = sycl_res.allocate<int>(N);
  sycl_res.memcpy(d_a, a, sizeof(int) * N);

  // _reductions_raja_hippolicy_start
  using EXEC_POL3   = RAJA::sycl_exec<SYCL_BLOCK_SIZE>;
  // _reductions_raja_hippolicy_end

  int sycl_sum = 0;
  int sycl_min = std::numeric_limits<int>::max();
  int sycl_max = std::numeric_limits<int>::min();
  VALLOC_INT sycl_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT sycl_maxloc(std::numeric_limits<int>::min(), -1);

  RAJA::forall<EXEC_POL3>(sycl_res, arange,
    RAJA::expt::Reduce<RAJA::operators::plus>(&sycl_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&sycl_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&sycl_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&sycl_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&sycl_maxloc),
    RAJA::expt::KernelName("RAJA Reduce SYCL Kernel"),
    [=] RAJA_DEVICE (int i, int &_sycl_sum, int &_sycl_min, int &_sycl_max, VALLOC_INT &_sycl_minloc, VALLOC_INT &_sycl_maxloc) {
      _sycl_sum += d_a[i];

      _sycl_min = RAJA_MIN(d_a[i], _sycl_min);
      _sycl_max = RAJA_MAX(d_a[i], _sycl_max);

      _sycl_minloc = RAJA_MIN(VALLOC_INT(d_a[i], i), _sycl_minloc);
      _sycl_maxloc = RAJA_MAX(VALLOC_INT(d_a[i], i), _sycl_maxloc);
      //_sycl_minloc.min(d_a[i], i);
      //_sycl_maxloc.max(d_a[i], i);
    }
  );

  std::cout << "\tsum = " << sycl_sum << std::endl;
  std::cout << "\tmin = " << sycl_min << std::endl;
  std::cout << "\tmax = " << sycl_max << std::endl;
  std::cout << "\tmin, loc = " << sycl_minloc.getVal() << " , "
                               << sycl_minloc.getLoc() << std::endl;
  std::cout << "\tmax, loc = " << sycl_maxloc.getVal() << " , "
                               << sycl_maxloc.getLoc() << std::endl;

  sycl_res.deallocate(d_a);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  host_res.deallocate(a);

  std::cout << "\n DONE!...\n";

  return 0;
}
