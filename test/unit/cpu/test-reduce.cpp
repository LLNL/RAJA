
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
/// Source file containing tests for RAJA loop reduction operations.
///

#include <time.h>
#include <cmath>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/util/defines.hpp"

#include "buildIndexSet.hpp"

using namespace RAJA;


using TestingTypes = ::testing::
    Types<
        std::tuple<ExecPolicy<seq_segit, simd_exec>, seq_reduce>
#ifdef RAJA_ENABLE_OPENMP
        ,
        std::tuple<ExecPolicy<omp_parallel_for_segit, simd_exec>, omp_reduce>,
        std::tuple<ExecPolicy<omp_parallel_for_segit, simd_exec>,
                   omp_reduce_ordered>
#endif
#ifdef RAJA_ENABLE_TBB
        ,
        std::tuple<ExecPolicy<seq_segit, tbb_for_exec>, tbb_reduce>,
        std::tuple<ExecPolicy<tbb_for_exec, simd_exec>, tbb_reduce>
#endif
        >;

template <typename Tuple>
class IndexSetReduce : public ::testing::Test
{
public:
  IndexSet iset;
  Index_type alen;
  RAJAVec<Index_type> is_indices;
  Real_ptr in_array;
  Real_ptr test_array;

  virtual void SetUp()
  {
    alen = buildIndexSet(&iset, static_cast<IndexSetBuildMethod>(0)) + 1;
    in_array = (Real_ptr)allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
    test_array =
        (Real_ptr)allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
    for (Index_type i = 0; i < alen; ++i) {
      in_array[i] = Real_type(rand() % 65536);
    }
    getIndices(is_indices, iset);
  }

  virtual void TearDown()
  {
    free_aligned(in_array);
    free_aligned(test_array);
  }
};

TYPED_TEST_CASE(IndexSetReduce, TestingTypes);

TYPED_TEST(IndexSetReduce, ReduceMinTest)
{
  using ISET_POLICY_T = typename std::tuple_element<0, TypeParam>::type;
  using REDUCE_POLICY_T = typename std::tuple_element<1, TypeParam>::type;

  for (Index_type i = 0; i < this->alen; ++i) {
    this->test_array[i] = fabs(this->in_array[i]);
  }

  const RAJA::Index_type ref_min_indx =
      this->is_indices[this->is_indices.size() / 2];
  const Real_type ref_min_val = -100.0;

  this->test_array[ref_min_indx] = ref_min_val;

  RAJA::ReduceMin<REDUCE_POLICY_T, Real_type> tmin0(1.0e+20);
  RAJA::ReduceMin<REDUCE_POLICY_T, Real_type> tmin1(-200.0);
  tmin1.min(-100.0);

  int loops = 2;

  for (int k = 1; k <= loops; ++k) {
    RAJA::forall<ISET_POLICY_T>(this->iset, [=](RAJA::Index_type idx) {
      tmin0.min(k * this->test_array[idx]);
      tmin1.min(this->test_array[idx]);
    });
    ASSERT_EQ(Real_type(tmin0), Real_type(k * ref_min_val));
    ASSERT_EQ(tmin1.get(), Real_type(-200.0));
  }
}


TYPED_TEST(IndexSetReduce, ReduceMinLocTest)
{
  using ISET_POLICY_T = typename std::tuple_element<0, TypeParam>::type;
  using REDUCE_POLICY_T = typename std::tuple_element<1, TypeParam>::type;

  for (Index_type i = 0; i < this->alen; ++i) {
    this->test_array[i] = fabs(this->in_array[i]);
  }

  const Index_type ref_min_indx =
      Index_type(this->is_indices[this->is_indices.size() / 2]);
  const Real_type ref_min_val = -100.0;

  this->test_array[ref_min_indx] = ref_min_val;

  ReduceMinLoc<REDUCE_POLICY_T, Real_type> tmin0(1.0e+20, -1);
  ReduceMinLoc<REDUCE_POLICY_T, Real_type> tmin1(-200.0, -1);
  tmin1.minloc(-100.0, -1);

  forallN<NestedPolicy<ExecList<ISET_POLICY_T>>>(
      this->iset, [=](Index_type idx) {
        tmin0.minloc(1 * this->test_array[idx], idx);
        tmin1.minloc(this->test_array[idx], idx);
      });

  ASSERT_EQ(tmin0.getLoc(), ref_min_indx);
  ASSERT_EQ(tmin1.getLoc(), -1);
  ASSERT_EQ(Real_type(tmin0), Real_type(1 * ref_min_val));
  ASSERT_EQ(tmin1.get(), Real_type(-200.0));

  forallN<NestedPolicy<ExecList<ISET_POLICY_T>>>(
      this->iset, [=](Index_type idx) {
        tmin0.minloc(2 * this->test_array[idx], idx);
        tmin1.minloc(this->test_array[idx], idx);
      });

  ASSERT_EQ(Real_type(tmin0), Real_type(2 * ref_min_val));
  ASSERT_EQ(tmin1.get(), Real_type(-200.0));
  ASSERT_EQ(tmin0.getLoc(), ref_min_indx);
  ASSERT_EQ(tmin1.getLoc(), -1);
}

TYPED_TEST(IndexSetReduce, ReduceMaxTest)
{
  using ISET_POLICY_T = typename std::tuple_element<0, TypeParam>::type;
  using REDUCE_POLICY_T = typename std::tuple_element<1, TypeParam>::type;

  for (Index_type i = 0; i < this->alen; ++i) {
    this->test_array[i] = -fabs(this->in_array[i]);
  }

  const Index_type ref_max_indx =
      Index_type(this->is_indices[this->is_indices.size() / 2]);
  const Real_type ref_max_val = 100.0;

  this->test_array[ref_max_indx] = ref_max_val;

  ReduceMax<REDUCE_POLICY_T, Real_type> tmax0(-1.0e+20);
  ReduceMax<REDUCE_POLICY_T, Real_type> tmax1(200.0);
  tmax1.max(100);

  int loops = 2;

  for (int k = 1; k <= loops; ++k) {

    forall<ISET_POLICY_T>(this->iset, [=](Index_type idx) {
      tmax0.max(k * this->test_array[idx]);
      tmax1.max(this->test_array[idx]);
    });

    ASSERT_EQ(Real_type(tmax0), Real_type(k * ref_max_val));
    ASSERT_EQ(tmax1.get(), Real_type(200.0));
  }
}

TYPED_TEST(IndexSetReduce, ReduceMaxLocTest)
{
  using ISET_POLICY_T = typename std::tuple_element<0, TypeParam>::type;
  using REDUCE_POLICY_T = typename std::tuple_element<1, TypeParam>::type;

  for (Index_type i = 0; i < this->alen; ++i) {
    this->test_array[i] = -fabs(this->in_array[i]);
  }

  const Index_type ref_max_indx =
      Index_type(this->is_indices[this->is_indices.size() / 2]);
  const Real_type ref_max_val = 100.0;

  this->test_array[ref_max_indx] = ref_max_val;

  ReduceMaxLoc<REDUCE_POLICY_T, Real_type> tmax0(-1.0e+20, -1);
  ReduceMaxLoc<REDUCE_POLICY_T, Real_type> tmax1(200.0, -1);
  tmax1.maxloc(100.0, -1);

  forall<ISET_POLICY_T>(this->iset, [=](Index_type idx) {
    tmax0.maxloc(1 * this->test_array[idx], idx);
    tmax1.maxloc(this->test_array[idx], idx);
  });

  ASSERT_EQ(tmax0.getLoc(), ref_max_indx);
  ASSERT_EQ(tmax1.getLoc(), -1);
  ASSERT_EQ(Real_type(tmax0), Real_type(1 * ref_max_val));
  ASSERT_EQ(tmax1.get(), Real_type(200.0));

  forall<ISET_POLICY_T>(this->iset, [=](Index_type idx) {
    tmax0.maxloc(2 * this->test_array[idx], idx);
    tmax1.maxloc(this->test_array[idx], idx);
  });

  ASSERT_EQ(Real_type(tmax0), Real_type(2 * ref_max_val));
  ASSERT_EQ(tmax1.get(), Real_type(200.0));
  ASSERT_EQ(tmax0.getLoc(), ref_max_indx);
  ASSERT_EQ(tmax1.getLoc(), -1);
}

TYPED_TEST(IndexSetReduce, ReduceSumTest)
{
  using ISET_POLICY_T = typename std::tuple_element<0, TypeParam>::type;
  using REDUCE_POLICY_T = typename std::tuple_element<1, TypeParam>::type;

  Real_type ref_sum = 0.0;

  for (size_t i = 0; i < this->is_indices.size(); ++i) {
    ref_sum += this->in_array[this->is_indices[i]];
  }

  ReduceSum<REDUCE_POLICY_T, Real_type> tsum0(0.0);
  ReduceSum<REDUCE_POLICY_T, Real_type> tsum1(5.0);
  tsum1 += 0.0;

  int loops = 2;

  for (int k = 1; k <= loops; ++k) {

    forall<ISET_POLICY_T>(this->iset, [=](Index_type idx) {
      tsum0 += this->in_array[idx];
      tsum1 += 1.0;
    });

    ASSERT_FLOAT_EQ(Real_type(tsum0), Real_type(k * ref_sum));
    ASSERT_FLOAT_EQ(tsum1.get(), Real_type(k * this->iset.getLength() + 5.0));
  }
}
