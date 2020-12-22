//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_LOC_MINEQSEG_HPP__
#define __TEST_KERNEL_LOC_MINEQSEG_HPP__

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename FORALL_POLICY, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelLocMinEqSegTestImpl(const int xdim, const int ydim)
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;

  DATA_TYPE min;
  DATA_TYPE sum = (DATA_TYPE)0;
  INDEX_TYPE minloc = (INDEX_TYPE)(-1);

  // square 2D array, xdim x ydim
  INDEX_TYPE array_length = xdim;

  allocateReduceLocTestData<DATA_TYPE>( array_length,
                                        work_res,
                                        &work_array,
                                        &check_array);

  // initializing  values
  RAJA::TypedRangeSegment<INDEX_TYPE> seg(0, xdim);
  RAJA::forall<RAJA::seq_exec>(seg, [=] (INDEX_TYPE zz)
  {
    check_array[zz] = zz;
  });

  check_array[xdim-1] = -1;

  work_res.memcpy(work_array, check_array, sizeof(DATA_TYPE) * array_length);

  RAJA::TypedRangeSegment<INDEX_TYPE> colrange0(1, xdim/5);
  RAJA::TypedRangeSegment<INDEX_TYPE> colrange1(xdim/3, xdim/2);
  RAJA::TypedRangeSegment<INDEX_TYPE> colrange2(xdim/2+1, xdim-2);

  RAJA::TypedIndexSet<RAJA::TypedRangeSegment<INDEX_TYPE>> cset;

  cset.push_back( colrange0 );
  cset.push_back( colrange1 );
  cset.push_back( colrange2 );

  RAJA::ReduceMinLoc<REDUCE_POLICY, DATA_TYPE, DATA_TYPE> minloc_reducer((DATA_TYPE)1024, 0);

  RAJA::kernel<EXEC_POLICY>(cset,
                           [=] RAJA_HOST_DEVICE (int c) {
                             minloc_reducer.minloc(work_array[c], c);
                           });

  // CPU answer
  min = array_length * 2;
  for ( int x = 0; x < xdim; ++x ) {
    DATA_TYPE val = check_array[x];

    sum += val;

    if (val < min) {
      min = val;
      minloc = x;
    }
  }

  INDEX_TYPE raja_loc = minloc_reducer.getLoc();
  DATA_TYPE raja_min = (DATA_TYPE)minloc_reducer.get();

  ASSERT_DOUBLE_EQ((DATA_TYPE)min, (DATA_TYPE)raja_min);
  ASSERT_EQ(minloc, raja_loc);

  deallocateReduceLocTestData<DATA_TYPE>( work_res,
                                          work_array,
                                          check_array);
}


TYPED_TEST_SUITE_P(KernelLocMinEqSegTest);
template <typename T>
class KernelLocMinEqSegTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelLocMinEqSegTest, LocMinEqSegKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using FORALL_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelLocMinEqSegTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelLocMinEqSegTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(1053, 1053);
  KernelLocMinEqSegTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(5101, 5101);
}

REGISTER_TYPED_TEST_SUITE_P(KernelLocMinEqSegTest,
                            LocMinEqSegKernel);

#endif  // __TEST_KERNEL_LOC_MINEQSEG_HPP__
