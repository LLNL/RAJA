//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU forall traversals.
///

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"

using namespace RAJA;
using namespace std;

#include "Compare.hpp"

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{

  //
  //  Build vector of integers for creating List segments.
  //
  default_random_engine gen;
  uniform_real_distribution<double> dist(0.0, 1.0);

  RAJAVec<Index_type> lindices;
  Index_type idx = 0;
  while (lindices.size() < 10000) {
    double dval = dist(gen);
    if (dval > 0.3) {
      lindices.push_back(idx);
    }
    idx++;
  }

  cout << "\n lindices: first, last = " << lindices[0] << " , "
       << lindices[lindices.size() - 1] << endl;

  //
  // Construct index set with mix of Range and List segments.
  //
  IndexSet iset;

  Index_type rbeg;
  Index_type rend;
  Index_type last_idx;
  Index_type lseg_len = lindices.size();
  RAJAVec<Index_type> lseg(lseg_len);
  std::vector<Index_type> lseg_vec(lseg_len);

  // Create empty Range segment
  rbeg = 1;
  rend = 1;
  iset.push_back(RangeSegment(rbeg, rend));
  last_idx = rend;

  cout << "\n last_idx = " << last_idx << endl;

  // Create Range segment
  rbeg = 1;
  rend = 15782;
  iset.push_back(RangeSegment(rbeg, rend));
  last_idx = rend;

  cout << "\n last_idx = " << last_idx << endl;

  // Create List segment
  for (Index_type i = 0; i < lseg_len; ++i) {
    lseg[i] = lindices[i] + last_idx + 3;
  }
  iset.push_back(ListSegment(&lseg[0], lseg_len));
  last_idx = lseg[lseg_len - 1];

  cout << "\n last_idx = " << last_idx << endl;

  // Create List segment using alternate ctor
  for (Index_type i = 0; i < lseg_len; ++i) {
    lseg_vec[i] = lindices[i] + last_idx + 3;
  }
  iset.push_back(ListSegment(lseg_vec));
  last_idx = lseg_vec[lseg_len - 1];

  cout << "\n last_idx = " << last_idx << endl;

  // Create Range segment
  rbeg = last_idx + 16;
  rend = rbeg + 20490;
  iset.push_back(RangeSegment(rbeg, rend));
  last_idx = rend;

  cout << "\n last_idx = " << last_idx << endl;

  // Create Range segment
  rbeg = last_idx + 4;
  rend = rbeg + 27595;
  iset.push_back(RangeSegment(rbeg, rend));
  last_idx = rend;

  cout << "\n last_idx = " << last_idx << endl;

  // Create List segment
  for (Index_type i = 0; i < lseg_len; ++i) {
    lseg[i] = lindices[i] + last_idx + 5;
  }
  iset.push_back(ListSegment(&lseg[0], lseg_len));
  last_idx = lseg[lseg_len - 1];

  cout << "\n last_idx = " << last_idx << endl;

  // Create Range segment
  rbeg = last_idx + 1;
  rend = rbeg + 32003;
  iset.push_back(RangeSegment(rbeg, rend));
  last_idx = rend;

  cout << "\n last_idx = " << last_idx << endl;

  // Create List segment using alternate ctor
  for (Index_type i = 0; i < lseg_len; ++i) {
    lseg_vec[i] = lindices[i] + last_idx + 7;
  }
  iset.push_back(ListSegment(lseg_vec));
  last_idx = lseg_vec[lseg_len - 1];

  cout << "\n last_idx = " << last_idx << endl;

  //
  // Collect actual indices in index set for testing.
  //
  RAJAVec<Index_type> is_indices;
  getIndices(is_indices, iset);


  ///////////////////////////////////////////////////////////////////////////
  //
  // Set up data and reference solution for tests...
  //
  ///////////////////////////////////////////////////////////////////////////

  const Index_type array_length = last_idx + 1;

  cout << "\n\n GPU Traversal tests: last_idx = " << last_idx << " ( "
       << array_length << " )\n\n"
       << endl;

  //
  // Allocate and initialize managed data arrays.
  //
  Real_ptr parent;
  cudaMallocManaged((void **)&parent,
                    sizeof(Real_type) * array_length,
                    cudaMemAttachGlobal);
  for (Index_type i = 0; i < array_length; ++i) {
    parent[i] = static_cast<Real_type>(rand() % 65536);
  }

  Real_ptr test_array;
  cudaMallocManaged((void **)&test_array,
                    sizeof(Real_type) * array_length,
                    cudaMemAttachGlobal);
  cudaMemset(test_array, 0, sizeof(Real_type) * array_length);

  Real_ptr ref_array;
  cudaMallocManaged((void **)&ref_array,
                    sizeof(Real_type) * array_length,
                    cudaMemAttachGlobal);
  cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);

  s_ntests_run = 0;
  s_ntests_passed = 0;


  ///
  /// Define thread block size for CUDA exec policy
  ///
  const size_t block_size = 256;


  ///////////////////////////////////////////////////////////////////////////
  //
  // Run forall tests ....
  //
  ///////////////////////////////////////////////////////////////////////////

  cout << "\n\n BEGIN RAJA::forall tests..." << endl;

  ///
  /// Run 0 length range traversal test for sanity check
  ///

  // Reset reference and results arrays
  cudaMemset(test_array, 0, sizeof(Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  for (Index_type i = 0; i < 0; ++i) {
    ref_array[i] = parent[i] * parent[i];
  }

  forall<cuda_exec<block_size> >(0, 0, [=] __device__(Index_type idx) {
    test_array[idx] = parent[idx] * parent[idx];
  });

  s_ntests_run++;
  if (!array_equal(ref_array, test_array, 0)) {
    cout << "\n TEST FAILURE " << endl;
#if 0
      cout << endl << endl;
      for (Index_type i = 0; i < is_indices.size(); ++i) {
         cout << "test_array[" << is_indices[i] << "] = "
                   << test_array[ is_indices[i] ]
                   << " ( " << ref_array[ is_indices[i] ] << " ) " << endl;
      }
      cout << endl;
#endif
  } else {
    s_ntests_passed++;
  }

  ///
  /// Run range traversal test in its simplest form for sanity check
  ///

  // Reset reference and results arrays
  cudaMemset(test_array, 0, sizeof(Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  for (Index_type i = 0; i < array_length; ++i) {
    ref_array[i] = parent[i] * parent[i];
  }

  forall<cuda_exec<block_size> >(0,
                                 array_length,
                                 [=] __device__(Index_type idx) {
                                   test_array[idx] = parent[idx] * parent[idx];
                                 });

  s_ntests_run++;
  if (!array_equal(ref_array, test_array, array_length)) {
    cout << "\n TEST FAILURE " << endl;
#if 0
      cout << endl << endl;
      for (Index_type i = 0; i < is_indices.size(); ++i) {
         cout << "test_array[" << is_indices[i] << "] = "
                   << test_array[ is_indices[i] ]
                   << " ( " << ref_array[ is_indices[i] ] << " ) " << endl;
      }
      cout << endl;
#endif
  } else {
    s_ntests_passed++;
  }

  ///
  /// Run traversal test with IndexSet containing multiple segments.
  ///

  // Reset reference and results arrays
  cudaMemset(test_array, 0, sizeof(Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  for (Index_type i = 0; i < is_indices.size(); ++i) {
    ref_array[is_indices[i]] = parent[is_indices[i]] * parent[is_indices[i]];
  }

  forall<IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(
      iset, [=] __device__(Index_type idx) {
        test_array[idx] = parent[idx] * parent[idx];
      });

  s_ntests_run++;
  if (!array_equal(ref_array, test_array, array_length)) {
    cout << "\n TEST FAILURE " << endl;
#if 0
      cout << endl << endl;
      for (Index_type i = 0; i < is_indices.size(); ++i) {
         cout << "test_array[" << is_indices[i] << "] = "
                   << test_array[ is_indices[i] ]
                   << " ( " << ref_array[ is_indices[i] ] << " ) " << endl;
      }
      cout << endl;
#endif
  } else {
    s_ntests_passed++;
  }

  cout << "\n END RAJA::forall tests..." << endl;

  ///////////////////////////////////////////////////////////////////////////
  //
  // Run forall_Icount tests....
  //
  ///////////////////////////////////////////////////////////////////////////

  cout << "\n\n BEGIN RAJA::forall_Icount tests..." << endl;

  ///
  /// Run range Icount test in its simplest form for sanity check
  ///

  // Reset reference and results arrays
  cudaMemset(test_array, 0, sizeof(Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  for (Index_type i = 0; i < array_length; ++i) {
    ref_array[i] = parent[i] * parent[i];
  }

  forall_Icount<cuda_exec<block_size> >(
      0, array_length, 0, [=] __device__(Index_type icount, Index_type idx) {
        test_array[icount] = parent[idx] * parent[idx];
      });

  s_ntests_run++;
  if (!array_equal(ref_array, test_array, array_length)) {
    cout << "\n TEST FAILURE " << endl;
    cout << endl << endl;
    for (Index_type i = 0; i < is_indices.size(); ++i) {
      cout << "test_array[" << is_indices[i]
           << "] = " << test_array[is_indices[i]] << " ( "
           << ref_array[is_indices[i]] << " ) " << endl;
    }
    cout << endl;
  } else {
    s_ntests_passed++;
  }

  ///
  /// Run Icount test with IndexSet containing multiple segments.
  ///

  // Reset reference and results arrays
  cudaMemset(test_array, 0, sizeof(Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  Index_type test_alen = is_indices.size();
  for (Index_type i = 0; i < test_alen; ++i) {
    ref_array[i] = parent[is_indices[i]] * parent[is_indices[i]];
  }

  forall_Icount<IndexSet::ExecPolicy<seq_segit, cuda_exec<block_size> > >(
      iset, [=] __device__(Index_type icount, Index_type idx) {
        test_array[icount] = parent[idx] * parent[idx];
      });

  s_ntests_run++;
  if (!array_equal(ref_array, test_array, test_alen)) {
    cout << "\n TEST FAILURE " << endl;
    cout << endl << endl;
    for (Index_type i = 0; i < test_alen; ++i) {
      cout << "test_array[" << is_indices[i]
           << "] = " << test_array[is_indices[i]] << " ( "
           << ref_array[is_indices[i]] << " ) " << endl;
    }
    cout << endl;
  } else {
    s_ntests_passed++;
  }

  cout << "\n END RAJA::forall_Icount tests..." << endl;

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cudaFree(parent);
  cudaFree(ref_array);
  cudaFree(test_array);

  cout << "\n RAJA GPU Traversal tests DONE!!! " << endl;

  return !(s_ntests_passed == s_ntests_run);
}
