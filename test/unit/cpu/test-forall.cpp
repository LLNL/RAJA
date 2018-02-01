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

#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA/policy/tbb/policy.hpp"
#include "gtest/gtest.h"

using namespace RAJA;
using namespace std;

#include "buildIndexSet.hpp"

template <typename ISET_POLICY_T>
class ForallTest : public ::testing::Test
{
protected:
  Real_ptr in_array;
  Index_type alen;
  UnitIndexSet iset;
  RAJAVec<Index_type> is_indices;
  Real_ptr test_array;
  Real_ptr ref_icount_array;
  Real_ptr ref_forall_array;

  virtual void SetUp()
  {
      // AddSegments chosen arbitrarily; index set equivalence is tested elsewhere
      alen = buildIndexSet(&iset, IndexSetBuildMethod::AddSegments) + 1;

      in_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

      for (Index_type i = 0; i < alen; ++i) {
        in_array[i] = Real_type(rand() % 65536);
      }

      getIndices(is_indices, iset);

      test_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
      ref_icount_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
      ref_forall_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

      for (Index_type i = 0; i < alen; ++i) {
          test_array[i] = 0.0;
          ref_forall_array[i] = 0.0;
          ref_icount_array[i] = 0.0;
      }

      for (size_t i = 0; i < is_indices.size(); ++i) {
          ref_forall_array[is_indices[i]] = in_array[is_indices[i]] * in_array[is_indices[i]];
      }

      for (size_t i = 0; i < is_indices.size(); ++i) {
          ref_icount_array[i] = in_array[is_indices[i]] * in_array[is_indices[i]];
      }
  }

  virtual void TearDown()
  {
      free_aligned(in_array);
      free_aligned(test_array);
      free_aligned(ref_icount_array);
      free_aligned(ref_forall_array);
  }
};

TYPED_TEST_CASE_P(ForallTest);

TYPED_TEST_P(ForallTest, BasicForall)
{
    forall<TypeParam>(this->iset, [=](Index_type idx) {
      this->test_array[idx] = this->in_array[idx] * this->in_array[idx];
    });

    for (Index_type i = 0; i < this->alen; ++i) {
        EXPECT_EQ(this->ref_forall_array[i], this->test_array[i]);
    }
}

TYPED_TEST_P(ForallTest, BasicForallIcount)
{
    forall_Icount<TypeParam>(this->iset, [=](Index_type icount, Index_type idx) {
      this->test_array[icount] = this->in_array[idx] * this->in_array[idx];
    });

    for (Index_type i = 0; i < this->alen; ++i) {
        EXPECT_EQ(this->ref_icount_array[i], this->test_array[i]);
    }
}

REGISTER_TYPED_TEST_CASE_P(ForallTest, BasicForall, BasicForallIcount);

using SequentialTypes = ::testing::Types<
    ExecPolicy<seq_segit, seq_exec>,
    ExecPolicy<seq_segit, loop_exec>,
    ExecPolicy<seq_segit, simd_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, ForallTest, SequentialTypes);


#if defined(RAJA_ENABLE_OPENMP)
using OpenMPTypes = ::testing::Types<
    ExecPolicy<seq_segit, omp_parallel_for_exec>,
    ExecPolicy<omp_parallel_for_segit, seq_exec>,
    ExecPolicy<omp_parallel_for_segit, loop_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, ForallTest, OpenMPTypes);
#endif

#if defined(RAJA_ENABLE_TBB)
using TBBTypes = ::testing::Types<
    ExecPolicy<seq_segit, tbb_for_exec>,
    ExecPolicy<tbb_for_exec, seq_exec>,
    ExecPolicy<tbb_for_exec, loop_exec>,
    ExecPolicy<seq_segit, tbb_for_dynamic>,
    ExecPolicy<tbb_for_dynamic, seq_exec>,
    ExecPolicy<tbb_for_dynamic, loop_exec>
    >;

INSTANTIATE_TYPED_TEST_CASE_P(TBB, ForallTest, TBBTypes);
#endif
