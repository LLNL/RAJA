//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_LOC_MIN2D_HPP__
#define __TEST_KERNEL_LOC_MIN2D_HPP__

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelLocMin2DTestImpl(const int xdim, const int ydim)
{
  DATA_TYPE ** array;
  DATA_TYPE * data;

  DATA_TYPE min;
  DATA_TYPE sum = (DATA_TYPE)0;
  INDEX_TYPE minlocx = (INDEX_TYPE)(-1);
  INDEX_TYPE minlocy = (INDEX_TYPE)(-1);

  // square 2 dimensional, xdim x ydim
  INDEX_TYPE array_length = xdim * ydim;

  array = new DATA_TYPE*[ydim];
  data = new DATA_TYPE[array_length];

  // set rows to point to data
  for ( int ii = 0; ii < ydim; ++ii ) {
    array[ii] = data + ii * ydim;
  }

  // setting data values
  int count = 0;
  for ( int ii = 0; ii < ydim; ++ii ) {
    for ( int jj = 0; jj < xdim; ++jj ) {
      array[ii][jj] = (DATA_TYPE)(count++);
    }
  }

  // extreme value at the end
  array[ydim-1][xdim-1] = (DATA_TYPE)(-1);

  min = array_length * 2;

  // CPU answer
  for (int y = 0; y < ydim; ++y) {
    for ( int x = 0; x < xdim; ++x ) {
      DATA_TYPE val = array[y][x];

      sum += val;

      if (val < min) {
        min = val;
        minlocx = x;
        minlocy = y;
      }
    }
  }

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, xdim);
  RAJA::RangeSegment rowrange(0, ydim);

  struct Index2D {
     RAJA::Index_type idx, idy;
     constexpr Index2D() : idx(-1), idy(-1) {}
     constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
  };

  RAJA::ReduceMinLoc<ReducePolicy, DATA_TYPE, Index2D> minloc_reducer((DATA_TYPE)1024, Index2D(0, 0));

  RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             minloc_reducer.minloc(array[r][c], Index2D(c, r));
                           });

  Index2D raja_loc = minloc_reducer.getLoc();
  DATA_TYPE raja_min = (DATA_TYPE)minloc_reducer.get();

  ASSERT_DOUBLE_EQ((DATA_TYPE)min, (DATA_TYPE)raja_min);
  ASSERT_EQ(minlocx, raja_loc.idx);
  ASSERT_EQ(minlocy, raja_loc.idy);

  delete[] array;
  delete[] data;
}


TYPED_TEST_SUITE_P(KernelLocMin2DTest);
template <typename T>
class KernelLocMin2DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelLocMin2DTest, LocMin2DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  KernelLocMin2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(10, 10);
  KernelLocMin2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(1053, 1053);
  KernelLocMin2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(20101, 20101);
}

REGISTER_TYPED_TEST_SUITE_P(KernelLocMin2DTest,
                            LocMin2DKernel);

#endif  // __TEST_KERNEL_LOC_MIN2D_HPP__
