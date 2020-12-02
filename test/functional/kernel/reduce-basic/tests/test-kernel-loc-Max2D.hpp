//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_LOC_MAX2D_HPP__
#define __TEST_KERNEL_LOC_MAX2D_HPP__

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename FORALL_POLICY, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelLocMax2DTestImpl(const int xdim, const int ydim)
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE ** workarr2D;
  DATA_TYPE ** checkarr2D;
  DATA_TYPE * work_array;
  DATA_TYPE * check_array;

  DATA_TYPE max;
  DATA_TYPE sum = (DATA_TYPE)0;
  INDEX_TYPE maxlocx = (INDEX_TYPE)(-1);
  INDEX_TYPE maxlocy = (INDEX_TYPE)(-1);

  // square 2D array, xdim x ydim
  INDEX_TYPE array_length = xdim * ydim;

  allocateReduceLocTestData<DATA_TYPE>( array_length,
                                        work_res,
                                        &work_array,
                                        &check_array);

  allocateReduceLocTestData<DATA_TYPE *>( ydim,
                                          work_res,
                                          &workarr2D,
                                          &checkarr2D);

  // set rows to point to check and work _arrays
  RAJA::TypedRangeSegment<INDEX_TYPE> seg(0,ydim);
  RAJA::forall<FORALL_POLICY>(seg, [=] RAJA_HOST_DEVICE(INDEX_TYPE zz)
  {
    workarr2D[zz] = work_array + zz * ydim;
  });

  RAJA::forall<RAJA::seq_exec>(seg, [=] (INDEX_TYPE zz)
  {
    checkarr2D[zz] = check_array + zz * ydim;
  });

  // initializing  values
  RAJA::forall<RAJA::seq_exec>(seg, [=] (INDEX_TYPE zz)
  {
    for ( int xx = 0; xx < xdim; ++xx )
    {
      checkarr2D[zz][xx] = zz*xdim + xx;
    }
    checkarr2D[ydim-1][xdim-1] = -1;
  });

  work_res.memcpy(work_array, check_array, sizeof(DATA_TYPE) * array_length);

  RAJA::TypedRangeSegment<INDEX_TYPE> colrange(0, xdim);
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange(0, ydim);

  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, Index2D> maxloc_reducer((DATA_TYPE)0, Index2D(0, 0));

  RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(colrange, rowrange),
                           [=] RAJA_HOST_DEVICE (int c, int r) {
                             maxloc_reducer.maxloc(workarr2D[r][c], Index2D(c, r));
                           });

  // CPU answer
  max = -2;
  for (int y = 0; y < ydim; ++y) {
    for ( int x = 0; x < xdim; ++x ) {
      DATA_TYPE val = checkarr2D[y][x];

      sum += val;

      if (val > max) {
        max = val;
        maxlocx = x;
        maxlocy = y;
      }
    }
  }

  Index2D raja_loc = maxloc_reducer.getLoc();
  DATA_TYPE raja_max = (DATA_TYPE)maxloc_reducer.get();

  ASSERT_DOUBLE_EQ((DATA_TYPE)max, (DATA_TYPE)raja_max);
  ASSERT_EQ(maxlocx, raja_loc.idx);
  ASSERT_EQ(maxlocy, raja_loc.idy);

  deallocateReduceLocTestData<DATA_TYPE>( work_res,
                                          work_array,
                                          check_array);

  deallocateReduceLocTestData<DATA_TYPE *>( work_res,
                                            workarr2D,
                                            checkarr2D);
}


TYPED_TEST_SUITE_P(KernelLocMax2DTest);
template <typename T>
class KernelLocMax2DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelLocMax2DTest, LocMax2DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using FORALL_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelLocMax2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelLocMax2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(1053, 1053);
  KernelLocMax2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(20101, 20101);
}

REGISTER_TYPED_TEST_SUITE_P(KernelLocMax2DTest,
                            LocMax2DKernel);

#endif  // __TEST_KERNEL_LOC_MAX2D_HPP__
