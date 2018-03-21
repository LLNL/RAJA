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

template <typename POLICY_T>
class ForallViewTest : public ::testing::Test
{
protected:
  Real_ptr arr;
  const Index_type alen = 100000;
  Real_type test_val = 0.123;

  virtual void SetUp()
  {
    arr = (Real_ptr) allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

    for (Index_type i = 0; i < alen; ++i) {
      arr[i] = Real_type(rand() % 65536);
    }
  }

  virtual void TearDown()
  {
    free_aligned(arr);
  }
};

TYPED_TEST_CASE_P(ForallViewTest);

TYPED_TEST_P(ForallViewTest, ForallViewLayout)
{
  const Index_type talen = this->alen;
  Real_ptr tarr = this->arr;
  Real_type ttest_val = this->test_val;

  const RAJA::Layout<1> my_layout(talen);
  RAJA::View<Real_type, RAJA::Layout<1> > view(tarr, my_layout);

  forall<TypeParam>(RAJA::RangeSegment(0, talen), [=](Index_type i) {
    view(i) = ttest_val;
  });

  for (Index_type i = 0; i < talen; ++i) {
    EXPECT_EQ(tarr[i], ttest_val);
  }
}

TYPED_TEST_P(ForallViewTest, ForallViewOffsetLayout)
{
  const Index_type talen = this->alen;
  Real_ptr tarr = this->arr;
  Real_type ttest_val = this->test_val;

  RAJA::OffsetLayout<1> my_layout = 
                        RAJA::make_offset_layout<1>({{1}}, {{talen+1}}); 
  RAJA::View<Real_type, RAJA::OffsetLayout<1> > view(tarr, my_layout);

  forall<TypeParam>(RAJA::RangeSegment(1, talen+1), [=](Index_type i) { 
    view(i) = ttest_val;
  });

  for (Index_type i = 0; i < talen; ++i) { 
    EXPECT_EQ(tarr[i], ttest_val);
  }
}

REGISTER_TYPED_TEST_CASE_P(ForallViewTest, ForallViewLayout, ForallViewOffsetLayout);

using SequentialTypes = ::testing::Types< seq_exec, loop_exec, simd_exec >;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, ForallViewTest, SequentialTypes);


#if defined(RAJA_ENABLE_OPENMP)
using OpenMPTypes = ::testing::Types< omp_parallel_for_exec >;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, ForallViewTest, OpenMPTypes);
#endif

#if defined(RAJA_ENABLE_TBB)
using TBBTypes = ::testing::Types< tbb_for_exec, tbb_for_dynamic >;

INSTANTIATE_TYPED_TEST_CASE_P(TBB, ForallViewTest, TBBTypes);
#endif
