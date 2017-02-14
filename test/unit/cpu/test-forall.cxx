/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include <cstdlib>

#include <string>

#include "RAJA/RAJA.hxx"
#include "gtest/gtest.h"

using namespace RAJA;
using namespace std;

#include "Compare.hxx"
#include "buildIndexSet.hxx"

using forall_types = ::testing::Types<
    std::tuple<seq_segit, seq_exec>,
    std::tuple<seq_segit, simd_exec>
#if defined(RAJA_ENABLE_OPENMP)    
    ,std::tuple<seq_segit, omp_parallel_for_exec>,
    std::tuple<omp_parallel_for_segit, seq_exec>,
    std::tuple<omp_parallel_for_segit, simd_exec>
#endif 
#if defined(RAJA_ENABLE_CILK)
    ,std::tuple<seq_segit, cilk_for_exec>,
    std::tuple<cilk_for_segit, seq_exec>,
    std::tuple<clik_for_segit, simd_exec>
#endif
>;

template <typename ISET_POLICY_T>
class ForallTest : public ::testing::Test
{
protected:
  Real_ptr in_array;
  Index_type alen;
  IndexSet iset;
  RAJAVec<Index_type> is_indices;
  
  virtual void SetUp()
  {
      // AddSegments chosen arbitrarily; index set equivalence is tested elsewhere
      alen = buildIndexSet(&iset, IndexSetBuildMethod::AddSegments) + 1;

      in_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

      for (Index_type i = 0; i < alen; ++i) {
        in_array[i] = Real_type(rand() % 65536);
      }

      getIndices(is_indices, iset);
  }

  virtual void TearDown()
  {
      free(in_array);
  }
    
};

TYPED_TEST_CASE_P(ForallTest);

TYPED_TEST_P(ForallTest, runBasicForallTest)
{
    using Iset_Policy = typename std::tuple_element<0, TypeParam>::type;
    using Execution_Policy = typename std::tuple_element<1, TypeParam>::type;
    using ISET_POLICY_T = IndexSet::ExecPolicy<Iset_Policy, Execution_Policy>;
    
    Real_ptr in_array = this->in_array;
    Index_type alen = this->alen;
    IndexSet iset = this->iset;
    RAJAVec<Index_type> is_indices = this->is_indices;

    Real_ptr test_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
    Real_ptr ref_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

    for (Index_type i = 0; i < alen; ++i) {
        test_array[i] = 0.0;
        ref_array[i] = 0.0;
    }

    for (size_t i = 0; i < is_indices.size(); ++i) {
      ref_array[is_indices[i]] = in_array[is_indices[i]] * in_array[is_indices[i]];
    }

    forall<ISET_POLICY_T>(iset, [=](Index_type idx) {
      test_array[idx] = in_array[idx] * in_array[idx];
    });

    for (Index_type i = 0; i < alen; ++i) {
        EXPECT_EQ(ref_array[i], test_array[i]);
    }

    free(test_array);
    free(ref_array);
}

TYPED_TEST_P(ForallTest, runBasicForallIcountTest)
{
    using Iset_Policy = typename std::tuple_element<0, TypeParam>::type;
    using Execution_Policy = typename std::tuple_element<1, TypeParam>::type;
    using ISET_POLICY_T = IndexSet::ExecPolicy<Iset_Policy, Execution_Policy>;
    
    Real_ptr in_array = this->in_array;
    Index_type alen = this->alen;
    IndexSet iset = this->iset;
    RAJAVec<Index_type> is_indices = this->is_indices;

    Real_ptr test_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
    Real_ptr ref_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

    for (Index_type i = 0; i < alen; ++i) {
        test_array[i] = 0.0;
        ref_array[i] = 0.0;
    }

    for (size_t i = 0; i < is_indices.size(); ++i) {
      ref_array[i] = in_array[is_indices[i]] * in_array[is_indices[i]];
    }

    forall_Icount<ISET_POLICY_T>(iset, [=](Index_type icount, Index_type idx) {
      test_array[icount] = in_array[idx] * in_array[idx];
    });

    for (Index_type i = 0; i < alen; ++i) {
        EXPECT_EQ(ref_array[i], test_array[i]);
    }

    free(test_array);
    free(ref_array);
}

REGISTER_TYPED_TEST_CASE_P(ForallTest, runBasicForallTest, runBasicForallIcountTest);

INSTANTIATE_TYPED_TEST_CASE_P(ForallTests, ForallTest, forall_types);

