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

#include "RAJA_gtest.hpp"

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment, RAJA::RangeStrideSegment>;

const size_t block_size = 256;

static UnitIndexSet iset;
static RAJA::Index_type array_length;
static RAJA::RAJAVec<RAJA::Index_type> is_indices;
static RAJA::Real_ptr parent, ref_array, test_array;

struct ForallCUDA : ::testing::Test {
  virtual void SetUp()
  {
    using namespace RAJA;
    using namespace std;

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

    //
    // Construct index set with mix of Range and List segments.
    //
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

    // Create Range segment
    rbeg = 1;
    rend = 15782;
    iset.push_back(RangeSegment(rbeg, rend));
    last_idx = rend;

    // Create List segment
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_idx + 3;
    }
    iset.push_back(ListSegment(&lseg[0], lseg_len));
    last_idx = lseg[lseg_len - 1];

    // Create List segment using alternate ctor
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg_vec[i] = lindices[i] + last_idx + 3;
    }
    iset.push_back(ListSegment(lseg_vec));
    last_idx = lseg_vec[lseg_len - 1];

    // Create Range segment
    rbeg = last_idx + 16;
    rend = rbeg + 20490;
    iset.push_back(RangeSegment(rbeg, rend));
    last_idx = rend;

    // Create Range segment
    rbeg = last_idx + 4;
    rend = rbeg + 27595;
    iset.push_back(RangeSegment(rbeg, rend));
    last_idx = rend;

    // Create List segment
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg[i] = lindices[i] + last_idx + 5;
    }
    iset.push_back(ListSegment(&lseg[0], lseg_len));
    last_idx = lseg[lseg_len - 1];

    // Create Range segment
    rbeg = last_idx + 1;
    rend = rbeg + 32003;
    iset.push_back(RangeSegment(rbeg, rend));
    last_idx = rend;

    // Create List segment using alternate ctor
    for (Index_type i = 0; i < lseg_len; ++i) {
      lseg_vec[i] = lindices[i] + last_idx + 7;
    }
    iset.push_back(ListSegment(lseg_vec));
    last_idx = lseg_vec[lseg_len - 1];

    //
    // Collect actual indices in index set for testing.
    //
    getIndices(is_indices, iset);

    ///////////////////////////////////////////////////////////////////////////
    //
    // Set up data and reference solution for tests...
    //
    ///////////////////////////////////////////////////////////////////////////

    array_length = last_idx + 1;
    //
    // Allocate and initialize managed data arrays.
    //

    Index_type max_size = (array_length > Index_type(is_indices.size()))
                              ? array_length
                              : is_indices.size();

    cudaMallocManaged((void **)&parent,
                      sizeof(Real_type) * max_size,
                      cudaMemAttachGlobal);
    for (Index_type i = 0; i < max_size; ++i) {
      parent[i] = static_cast<Real_type>(rand() % 65536);
    }

    cudaMallocManaged((void **)&test_array,
                      sizeof(Real_type) * max_size,
                      cudaMemAttachGlobal);
    cudaMemset(test_array, 0, sizeof(Real_type) * max_size);

    cudaMallocManaged((void **)&ref_array,
                      sizeof(Real_type) * max_size,
                      cudaMemAttachGlobal);
    cudaMemset(ref_array, 0, sizeof(Real_type) * max_size);
  }

  virtual void TearDown()
  {
    cudaFree(::test_array);
    cudaFree(::ref_array);
    cudaFree(::parent);
    ::iset = UnitIndexSet();
    ::is_indices = RAJA::RAJAVec<RAJA::Index_type>();
  }
};

///
/// Run traversal with simple range-based iteration
///
CUDA_TEST_F(ForallCUDA, forall_range)
{
  RAJA::Real_ptr parent = ::parent;
  RAJA::Real_ptr test_array = ::test_array;
  RAJA::Real_ptr ref_array = ::ref_array;

  cudaMemset(test_array, 0, sizeof(RAJA::Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(RAJA::Real_type) * array_length);

  for (RAJA::Index_type i = 0; i < array_length; ++i) {
    ref_array[i] = parent[i] * parent[i];
  }

  RAJA::forall<RAJA::cuda_exec<block_size>>(
      RAJA::make_range(0, array_length), [=] __device__(RAJA::Index_type idx) {
        test_array[idx] = parent[idx] * parent[idx];
      });

  for (RAJA::Index_type i = 0; i < array_length; ++i) {
    ASSERT_FLOAT_EQ(ref_array[i], test_array[i]);
  }
}

///
/// Run range Icount test in its simplest form for sanity check
///
CUDA_TEST_F(ForallCUDA, forall_icount_range)
{
  RAJA::Real_ptr parent = ::parent;
  RAJA::Real_ptr test_array = ::test_array;
  RAJA::Real_ptr ref_array = ::ref_array;

  cudaMemset(test_array, 0, sizeof(RAJA::Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(RAJA::Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  for (RAJA::Index_type i = 0; i < array_length; ++i) {
    ref_array[i] = parent[i] * parent[i];
  }

  RAJA::forall_Icount<RAJA::cuda_exec<block_size>>(
      RAJA::make_range(0, array_length),
      0,
      [=] __device__(RAJA::Index_type icount, RAJA::Index_type idx) {
        test_array[icount] = parent[idx] * parent[idx];
      });

  for (RAJA::Index_type i = 0; i < array_length; ++i) {
    ASSERT_FLOAT_EQ(ref_array[i], test_array[i]);
  }
}

///
/// Run traversal test with IndexSet containing multiple segments.
///
CUDA_TEST_F(ForallCUDA, forall_indexset)
{
  RAJA::Real_ptr parent = ::parent;
  RAJA::Real_ptr test_array = ::test_array;
  RAJA::Real_ptr ref_array = ::ref_array;

  cudaMemset(test_array, 0, sizeof(RAJA::Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(RAJA::Real_type) * array_length);

  //
  // Generate reference result to check correctness.
  // Note: Reference does not use RAJA!!!
  //
  for (decltype(is_indices.size()) i = 0; i < is_indices.size(); ++i) {
    ref_array[is_indices[i]] = parent[is_indices[i]] * parent[is_indices[i]];
  }

  RAJA::forall<RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<block_size>>>(
      iset, [=] __device__(RAJA::Index_type idx) {
        test_array[idx] = parent[idx] * parent[idx];
      });

  for (RAJA::Index_type i = 0; i < array_length; ++i) {
    ASSERT_FLOAT_EQ(ref_array[i], test_array[i]);
  }
}

///
/// Run Icount test with IndexSet containing multiple segments.
///
CUDA_TEST_F(ForallCUDA, forall_icount_indexset)
{
  RAJA::Real_ptr parent = ::parent;
  RAJA::Real_ptr test_array = ::test_array;
  RAJA::Real_ptr ref_array = ::ref_array;

  cudaMemset(test_array, 0, sizeof(RAJA::Real_type) * array_length);
  cudaMemset(ref_array, 0, sizeof(RAJA::Real_type) * array_length);

  RAJA::Index_type test_alen = is_indices.size();
  for (RAJA::Index_type i = 0; i < test_alen; ++i) {
    ref_array[i] = parent[is_indices[i]] * parent[is_indices[i]];
  }

  RAJA::forall_Icount<RAJA::ExecPolicy<RAJA::seq_segit,
                                       RAJA::cuda_exec<block_size>>>(
      iset, [=] __device__(RAJA::Index_type icount, RAJA::Index_type idx) {
        test_array[icount] = parent[idx] * parent[idx];
      });

  for (RAJA::Index_type i = 0; i < array_length; ++i) {
    ASSERT_FLOAT_EQ(ref_array[i], test_array[i]);
  }
}
