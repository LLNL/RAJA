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

class ForallTest : public ::testing::Test
{
private:
  Real_ptr in_array;
  Index_type alen;
  IndexSet iset;
  RAJAVec<Index_type> is_indices;

protected:
  
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

  template <typename ISET_POLICY_T>
  void runBasicForallTest()
  {
      Real_ptr test_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
      Real_ptr ref_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

      std::fill(&test_array[0], &test_array[alen], 0.0);
      std::fill(&ref_array[0], &ref_array[alen], 0.0);

      for (Index_type i = 0; i < is_indices.size(); ++i) {
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
    
};

TEST_F(ForallTest, seq_segit_seq_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<seq_segit, seq_exec> >();
}

TEST_F(ForallTest, seq_segit_simd_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<seq_segit, simd_exec> >();
}

#if defined(RAJA_ENABLE_OPENMP)
TEST_F(ForallTest, seq_segit_omp_parallel_for_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >();
}

TEST_F(ForallTest, omp_parallel_for_segit_seq_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >();
}

TEST_F(ForallTest, omp_parallel_for_segit_simd_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> >();
}  
#endif // defined(RAJA_ENABLE_OPENMP)

#if defined(RAJA_ENABLE_CILK)
TEST_F(ForallTest, seq_segit_clik_for_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<seq_segit, cilk_for_exec> >();
}

TEST_F(ForallTest, cilk_for_segit_seq_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<cilk_for_segit, seq_exec> >();
}

TEST_F(ForallTest, cilk_for_segit_simd_exec)
{
  runBasicForallTest<IndexSet::ExecPolicy<cilk_for_segit, simd_exec> >();
}
#endif // defined(RAJA_ENABLE_CILK)  

class ForallIcountTest : public ::testing::Test
{
private:
  Real_ptr in_array;
  Index_type alen;
  IndexSet iset;
  RAJAVec<Index_type> is_indices;

protected:
  
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

  template <typename ISET_POLICY_T>
  void runBasicForallIcountTest()
  {
      Real_ptr test_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
      Real_ptr ref_array = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

      std::fill(&test_array[0], &test_array[alen], 0.0);
      std::fill(&ref_array[0], &ref_array[alen], 0.0);

      for (Index_type i = 0; i < is_indices.size(); ++i) {
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
    
};

TEST_F(ForallIcountTest, seq_segit_seq_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<seq_segit, seq_exec> >();
}

TEST_F(ForallIcountTest, seq_segit_simd_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<seq_segit, simd_exec> >();
}

#if defined(RAJA_ENABLE_OPENMP)
TEST_F(ForallIcountTest, seq_segit_omp_parallel_for_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec> >();
}

TEST_F(ForallIcountTest, omp_parallel_for_segit_seq_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec> >();
}

TEST_F(ForallIcountTest, omp_parallel_for_segit_simd_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> >();
}  
#endif // defined(RAJA_ENABLE_OPENMP)

#if defined(RAJA_ENABLE_CILK)
TEST_F(ForallIcountTest, seq_segit_clik_for_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<seq_segit, cilk_for_exec> >();
}

TEST_F(ForallIcountTest, cilk_for_segit_seq_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<cilk_for_segit, seq_exec> >();
}

TEST_F(ForallIcountTest, cilk_for_segit_simd_exec)
{
  runBasicForallIcountTest<IndexSet::ExecPolicy<cilk_for_segit, simd_exec> >();
}
#endif // defined(RAJA_ENABLE_CILK)  
