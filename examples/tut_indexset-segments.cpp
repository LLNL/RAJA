//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

/*
 *  Index sets and Segments Example
 *
 *  This example uses the daxpy kernel from a previous example. It
 *  illustrates how to use RAJA index sets and segments. This is 
 *  important for applications and algorithms that need to use 
 *  indirection arrays for irregular access. Combining range and
 *  list segments in a single index set, when possible, can 
 *  increase performance by allowing compilers to optimize for
 *  specific segment types (e.g., SIMD for range segments).
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment 
 *    -  Index list segment 
 *    -  Strided index range segment 
 *    -  TypedIndexSet segment container
 *    -  Hierarchical execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

//----------------------------------------------------------------------------//
// Define types for ListSegments and indices used in examples
//----------------------------------------------------------------------------//
// _raja_list_segment_type_start
using IdxType = RAJA::Index_type;
using ListSegType = RAJA::TypedListSegment<IdxType>;
// _raja_list_segment_type_end

//
// Functions to check and print results
//
void checkResult(double* v1, double* v2, IdxType len);
void printResult(double* v, int len);
 
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA index sets and segments example...\n";

//
// Define vector length
//
  const IdxType N = 1000000;

//
// Allocate and initialize vector data
//
  double* a0 = memoryManager::allocate<double>(N);
  double* aref = memoryManager::allocate<double>(N);

  double* a = memoryManager::allocate<double>(N);
  double* b = memoryManager::allocate<double>(N);
  
  double c = 3.14159;
  
  for (IdxType i = 0; i < N; i++) {
    a0[i] = 1.0;
    b[i] = 2.0;
  }


//----------------------------------------------------------------------------//
//
// C-version of the daxpy kernel to set the reference result.
//
  std::cout << "\n Running C-version of daxpy to set reference result...\n";

  std::memcpy( aref, a0, N * sizeof(double) );

// _csytle_daxpy_start
  for (IdxType i = 0; i < N; i++) {
    aref[i] += b[i] * c;
  }
// _csytle_daxpy_end

//printResult(a, N);

//----------------------------------------------------------------------------//
//
// In the following, we show RAJA versions of the daxpy operation and 
// using different Segment constructs and TypedIndexSets. These are all 
// run sequentially. The only thing that changes in these versions is
// the object passed to the 'forall' method that defines the iteration
// space.
//
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA range segment daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  // _rajaseq_daxpy_range_start
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _rajaseq_daxpy_range_end

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//
// Resource object used to construct list segment objects with indices
// living in host (CPU) memory.

  camp::resources::Resource host_res{camp::resources::Host()};


//
// RAJA list segment version #1
//
  std::cout << "\n Running RAJA list segment daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );  

//
// Collect indices in a vector to create list segment
//
// 
  // _rajaseq_daxpy_list_start
  std::vector<IdxType> idx;
  for (IdxType i = 0; i < N; ++i) {
    idx.push_back(i); 
  } 

  ListSegType idx_list( &idx[0], idx.size(), host_res );

  RAJA::forall<RAJA::seq_exec>(idx_list, [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _rajaseq_daxpy_list_end

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//
//
// RAJA list segment version #2
//
  std::cout << "\n Running RAJA list segment daxpy with indices reversed...\n";

  std::memcpy( a, a0, N * sizeof(double) );  

//
// Reverse the order of indices in the vector
//
  // _raja_list_segment_daxpy_reverse_start
  std::reverse( idx.begin(), idx.end() ); 

  ListSegType idx_reverse_list( &idx[0], idx.size(), host_res );

  RAJA::forall<RAJA::seq_exec>(idx_reverse_list, [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _raja_list_segment_daxpy_reverse_end

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//
//
// Alternatively, we can also use a RAJA strided range segment to run the
// loop in reverse.
//
  std::cout << "\n Running RAJA daxpy with indices reversed via negatively strided range segment...\n";

  std::memcpy( a, a0, N * sizeof(double) );

//
// Reverse the order of indices in the vector
//
  // _raja_range_segment_daxpy_negstride_start
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeStrideSegment(N-1, -1, -1), 
    [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _raja_range_segment_daxpy_negstride_end

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//

//
// Sequential index set execution policy used in several of the following
// example implementations.
//

  // _raja_seq_indexset_policy_daxpy_start
  using SEQ_ISET_EXECPOL = RAJA::ExecPolicy<RAJA::seq_segit,
                                            RAJA::seq_exec>;
  // _raja_seq_indexset_policy_daxpy_end

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (ListSegment) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  // _raja_indexset_list_daxpy_start
  RAJA::TypedIndexSet<ListSegType> is1;

  is1.push_back( idx_list );  // use list segment created earlier.
  
  RAJA::forall<SEQ_ISET_EXECPOL>(is1, [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _raja_indexset_list_daxpy_end

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (2 RangeSegments) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  // _raja_indexset_2ranges_daxpy_start
  RAJA::TypedIndexSet<RAJA::RangeSegment> is2;
  is2.push_back( RAJA::RangeSegment(0, N/2) );
  is2.push_back( RAJA::RangeSegment(N/2, N) );
  
  RAJA::forall<SEQ_ISET_EXECPOL>(is2, [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _raja_indexset_2ranges_daxpy_end

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  // _raja_indexset_2ranges_1list_daxpy_start
//
// Collect indices in a vector to create list segment
//
  std::vector<IdxType> idx1;
  for (IdxType i = N/3; i < 2*N/3; ++i) {
    idx1.push_back(i);
  }

  ListSegType idx1_list( &idx1[0], idx1.size(), host_res );

  RAJA::TypedIndexSet<RAJA::RangeSegment, ListSegType> is3;
  is3.push_back( RAJA::RangeSegment(0, N/3) );
  is3.push_back( idx1_list );
  is3.push_back( RAJA::RangeSegment(2*N/3, N) );
 
  RAJA::forall<SEQ_ISET_EXECPOL>(is3, [=] (IdxType i) {
    a[i] += b[i] * c;
  });
  // _raja_indexset_2ranges_1list_daxpy_end

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
//
// Run the previous version in parallel (2 different ways) just for fun...
//

  std::cout << 
    "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy\n" << 
    " (sequential iteration over segments, OpenMP parallel segment execution)...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  // _raja_indexset_ompinnerpolicy_daxpy_start
  using OMP_ISET_EXECPOL1 = RAJA::ExecPolicy<RAJA::seq_segit,
                                             RAJA::omp_parallel_for_exec>;
  // _raja_indexset_ompinnerpolicy_daxpy_end

  RAJA::forall<OMP_ISET_EXECPOL1>(is3, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << 
    "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy\n" << 
    " (OpenMP parallel iteration over segments, sequential segment execution)...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  // _raja_indexset_ompouterpolicy_daxpy_start
  using OMP_ISET_EXECPOL2 = RAJA::ExecPolicy<RAJA::omp_parallel_for_segit,
                                             RAJA::seq_exec>;
  // _raja_indexset_ompouterpolicy_daxpy_end

  RAJA::forall<OMP_ISET_EXECPOL2>(is3, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

//
// We create a new resource object and index set so that list segment 
// indices live in CUDA deviec memory.
//
  camp::resources::Resource cuda_res{camp::resources::Cuda()};

  ListSegType idx1_list_cuda( &idx1[0], idx1.size(), cuda_res );

  RAJA::TypedIndexSet<RAJA::RangeSegment, ListSegType> is3_cuda;
  is3_cuda.push_back( RAJA::RangeSegment(0, N/3) );
  is3_cuda.push_back( idx1_list_cuda );
  is3_cuda.push_back( RAJA::RangeSegment(2*N/3, N) );


  std::cout << 
    "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy\n" << 
    " (sequential iteration over segments, CUDA parallel segment execution)...\n";

  // _raja_indexset_cudapolicy_daxpy_start
  using CUDA_ISET_EXECPOL = RAJA::ExecPolicy<RAJA::seq_segit,
                                             RAJA::cuda_exec<CUDA_BLOCK_SIZE>>;
  // _raja_indexset_cudapolicy_daxpy_end

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::forall<CUDA_ISET_EXECPOL>(is3_cuda, [=] RAJA_DEVICE (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

//
// We create a new resource object and index set so that list segment
// indices live in Hip deviec memory.
//
  camp::resources::Resource hip_res{camp::resources::Hip()};

  ListSegType idx1_list_hip( &idx1[0], idx1.size(), hip_res );

  RAJA::TypedIndexSet<RAJA::RangeSegment, ListSegType> is3_hip;
  is3_hip.push_back( RAJA::RangeSegment(0, N/3) );
  is3_hip.push_back( idx1_list_hip );
  is3_hip.push_back( RAJA::RangeSegment(2*N/3, N) );

  std::cout <<
    "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy\n" <<
    " (sequential iteration over segments, HIP parallel segment execution)...\n";

  // _raja_indexset_hippolicy_daxpy_start
  using HIP_ISET_EXECPOL = RAJA::ExecPolicy<RAJA::seq_segit,
                                            RAJA::hip_exec<HIP_BLOCK_SIZE>>;
  // _raja_indexset_hippolicy_daxpy_end

  double* d_a = memoryManager::allocate_gpu<double>(N);
  double* d_b = memoryManager::allocate_gpu<double>(N);

  hipErrchk(hipMemcpy( d_a, a0, N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_b,  b, N * sizeof(double), hipMemcpyHostToDevice ));

  RAJA::forall<HIP_ISET_EXECPOL>(is3_hip, [=] RAJA_DEVICE (IdxType i) {
    d_a[i] += d_b[i] * c;
  });

  hipErrchk(hipMemcpy( a, d_a, N * sizeof(double), hipMemcpyDeviceToHost ));

  checkResult(a, aref, N);
//printResult(a, N);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
#endif

//----------------------------------------------------------------------------//

//
// Clean up. 
//
  memoryManager::deallocate(a); 
  memoryManager::deallocate(b); 
  memoryManager::deallocate(a0); 
  memoryManager::deallocate(aref); 
 
  std::cout << "\n DONE!...\n";
 
  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(double* v1, double* v2, IdxType len) 
{
  bool match = true;
  for (IdxType i = 0; i < len; i++) {
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
void printResult(double* v, IdxType len) 
{
  std::cout << std::endl;
  for (IdxType i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
} 

