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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

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
 *    -  IndexSet segment container
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

//----------------------------------------------------------------------------//
// Define types for ListSegments and indices used in examples
//----------------------------------------------------------------------------//
using IdxType = RAJA::Index_type;
using ListSegType = RAJA::TypedListSegment<IdxType>;

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

  for (IdxType i = 0; i < N; i++) {
    aref[i] += b[i] * c;
  }

//printResult(a, N);

//----------------------------------------------------------------------------//
//
// In the following, we show RAJA versions of the daxpy operation and 
// using different Segment constructs and IndexSets. These are all 
// run sequentially. The only thing that changes in these versions is
// the object passed to the 'forall' method that defines the iteration
// space.
//
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA range segment daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//
//
// RAJA list segment version #1
//
  std::cout << "\n Running RAJA list segment daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );  

//
// Collect indices in a vector to create list segment
//
  std::vector<IdxType> idx;
  for (IdxType i = 0; i < N; ++i) {
    idx.push_back(i); 
  } 

  ListSegType idx_list( &idx[0], idx.size() );

  RAJA::forall<RAJA::seq_exec>(idx_list, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

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
  std::reverse( idx.begin(), idx.end() ); 

  ListSegType idx_reverse_list( &idx[0], idx.size() );

  RAJA::forall<RAJA::seq_exec>(idx_reverse_list, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

//
// Sequential index set execution policy used in several of the following
// example implementations.
//

  using SEQ_ISET_EXECPOL = RAJA::ExecPolicy<RAJA::seq_segit,
                                            RAJA::seq_exec>;

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (ListSegment) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::TypedIndexSet<ListSegType> is1;

  is1.push_back( idx_list );  // use list segment created earlier.
  
  RAJA::forall<SEQ_ISET_EXECPOL>(is1, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (2 RangeSegments) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::TypedIndexSet<RAJA::RangeSegment> is2;
  is2.push_back( RAJA::RangeSegment(0, N/2) );
  is2.push_back( RAJA::RangeSegment(N/2, N) );
  
  RAJA::forall<SEQ_ISET_EXECPOL>(is2, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

//
// Collect indices in a vector to create list segment
//
  std::vector<IdxType> idx1;
  for (IdxType i = N/3; i < 2*N/3; ++i) {
    idx1.push_back(i);
  }

  ListSegType idx1_list( &idx1[0], idx1.size() );

  RAJA::TypedIndexSet<RAJA::RangeSegment, ListSegType> is3;
  is3.push_back( RAJA::RangeSegment(0, N/3) );
  is3.push_back( idx1_list );
  is3.push_back( RAJA::RangeSegment(2*N/3, N) );
 
  RAJA::forall<SEQ_ISET_EXECPOL>(is3, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

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

  using OMP_ISET_EXECPOL1 = RAJA::ExecPolicy<RAJA::seq_segit,
                                             RAJA::omp_parallel_for_exec>;

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

  using OMP_ISET_EXECPOL2 = RAJA::ExecPolicy<RAJA::omp_parallel_for_segit,
                                               RAJA::seq_exec>;

  RAJA::forall<OMP_ISET_EXECPOL2>(is3, [=] (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << 
    "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy\n" << 
    " (sequential iteration over segments, CUDA parallel segment execution)...\n";

  using OMP_ISET_EXECPOL3 = RAJA::ExecPolicy<RAJA::seq_segit,
                                             RAJA::cuda_exec<CUDA_BLOCK_SIZE>>;

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::forall<OMP_ISET_EXECPOL3>(is3, [=] RAJA_DEVICE (IdxType i) {
    a[i] += b[i] * c;
  });

  checkResult(a, aref, N);
//printResult(a, N);
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

