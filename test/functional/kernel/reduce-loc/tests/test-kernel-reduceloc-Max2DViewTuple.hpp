//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REDUCELOC_MAX2DVIEWTUPLE_HPP__
#define __TEST_KERNEL_REDUCELOC_MAX2DVIEWTUPLE_HPP__

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename FORALL_POLICY, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelLocMax2DViewTupleTestImpl(const int xdim, const int ydim)
{
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE ** workarr2D;
  DATA_TYPE ** checkarr2D;
  DATA_TYPE ** testarr2D;
  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  // square 2D array, xdim x ydim
  INDEX_TYPE array_length = xdim * ydim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  allocateForallTestData<DATA_TYPE *> ( ydim,
                                        work_res,
                                        &workarr2D,
                                        &checkarr2D,
                                        &testarr2D
                                      );

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
    checkarr2D[ydim-1][xdim-1] = 0;
  });

  work_res.memcpy(work_array, check_array, sizeof(DATA_TYPE) * array_length);

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  //#pragma omp target data map(to:work_array[0:array_length])
#endif

  RAJA::TypedRangeSegment<INDEX_TYPE> colrange(0, xdim);
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange(0, ydim);

  RAJA::View<DATA_TYPE, RAJA::Layout<2>> ArrView(work_array, xdim, ydim);

  RAJA::tuple<DATA_TYPE, DATA_TYPE> LocTup(0, 0);

  RAJA::ReduceMaxLoc<REDUCE_POLICY, DATA_TYPE, RAJA::tuple<DATA_TYPE, DATA_TYPE>> maxloc_reducer((DATA_TYPE)0, LocTup);

  RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(colrange, rowrange),
                           [=] RAJA_HOST_DEVICE (int c, int r) {
                             maxloc_reducer.maxloc(ArrView(r, c), RAJA::make_tuple((DATA_TYPE)c, (DATA_TYPE)r));
                           });

  // CPU answer
  RAJA::ReduceMaxLoc<RAJA::seq_reduce, DATA_TYPE, Index2D> checkmaxloc_reducer((DATA_TYPE)0, Index2D(0, 0));

  RAJA::forall<RAJA::seq_exec>(colrange, [=] (INDEX_TYPE c) {
    for( int r = 0; r < ydim; ++r)
    {
      checkmaxloc_reducer.maxloc(checkarr2D[r][c], Index2D(c, r));
    }
  });

  RAJA::tuple<DATA_TYPE, DATA_TYPE> raja_loc = maxloc_reducer.getLoc();
  DATA_TYPE raja_max = (DATA_TYPE)maxloc_reducer.get();
  Index2D checkraja_loc = checkmaxloc_reducer.getLoc();
  DATA_TYPE checkraja_max = (DATA_TYPE)checkmaxloc_reducer.get();

  ASSERT_DOUBLE_EQ((DATA_TYPE)checkraja_max, (DATA_TYPE)raja_max);
  ASSERT_EQ(checkraja_loc.idx, RAJA::get<0>(raja_loc));
  ASSERT_EQ(checkraja_loc.idy, RAJA::get<1>(raja_loc));

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );

  deallocateForallTestData<DATA_TYPE *> ( work_res,
                                          workarr2D,
                                          checkarr2D,
                                          testarr2D
                                        );
}


TYPED_TEST_SUITE_P(KernelLocMax2DViewTupleTest);
template <typename T>
class KernelLocMax2DViewTupleTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelLocMax2DViewTupleTest, LocMax2DViewTupleKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using FORALL_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<5>>::type;

  KernelLocMax2DViewTupleTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelLocMax2DViewTupleTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(151, 151);
  KernelLocMax2DViewTupleTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, FORALL_POLICY, EXEC_POLICY, REDUCE_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelLocMax2DViewTupleTest,
                            LocMax2DViewTupleKernel);

#endif  // __TEST_KERNEL_REDUCELOC_MAX2DVIEWTUPLE_HPP__
