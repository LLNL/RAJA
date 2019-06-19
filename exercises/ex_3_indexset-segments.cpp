//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
 *    -  IndexSet segment container
 *    -  Hierarchical execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

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
  
  //TODO: RAJA variant using a range segment

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//
//
// RAJA list segment version 
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

  //TODO: RAJA variant using a list segment

  checkResult(a, aref, N);
//printResult(a, N);

//----------------------------------------------------------------------------//

//
// Sequential index set execution policy used in several of the following
// example implementations.
//

  //TODO: Create a sequential index set execution policy

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (ListSegment) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::TypedIndexSet<ListSegType> is1;
  
   // TODO: RAJA variant using an index set
   //        Use the list segment created earlier
  

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (2 RangeSegments) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

  RAJA::TypedIndexSet<RAJA::RangeSegment> is2;

  // TODO: RAJA variant using an index set
  // Create an index set composed of two range segments

  checkResult(a, aref, N);
//printResult(a, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA index set (2 RangeSegments, 1 ListSegment) daxpy...\n";

  std::memcpy( a, a0, N * sizeof(double) );

//
// Collect indices in a vector to create list segment
//
  std::vector<IdxType> idx1;

  ListSegType idx1_list( &idx1[0], idx1.size() );

  RAJA::TypedIndexSet<RAJA::RangeSegment, ListSegType> is3;
  
  //TODO: RAJA variant using index set
  //      Create an index set composed of range segments
  //      and list segment types


  checkResult(a, aref, N);
//printResult(a, N);


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

