//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_DYNAMIC2D_HPP__
#define __TEST_KERNEL_TILE_DYNAMIC2D_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelTileDynamic2DTestImpl(const int rows, const int cols)
{
  // This test emulates matrix transposition with tiling.

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  // holds transposed matrices
  DATA_TYPE * work_array_t;
  DATA_TYPE * check_array_t;
  DATA_TYPE * test_array_t;

  INDEX_TYPE array_length = rows * cols;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array_t,
                                      &check_array_t,
                                      &test_array_t
                                    );

  RAJA::View<DATA_TYPE, RAJA::Layout<2>> HostView( test_array, rows, cols );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> HostTView( test_array_t, cols, rows );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> WorkView( work_array, rows, cols );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> WorkTView( work_array_t, cols, rows );
  RAJA::View<DATA_TYPE, RAJA::Layout<2>> CheckTView( check_array_t, cols, rows );

  // initialize arrays
  std::iota( test_array, test_array + array_length, 1 );
  std::iota( test_array_t, test_array_t + array_length, 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * array_length );
  work_res.memcpy( work_array_t, test_array_t, sizeof(DATA_TYPE) * array_length );

  // transpose test_array on CPU
  for ( int rr = 0; rr < rows; ++rr )
  {
    for ( int cc = 0; cc < cols; ++cc )
    {
      HostTView( cc, rr ) = HostView( rr, cc ); 
    }
  }

  // transpose work_array
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, rows );
  RAJA::TypedRangeSegment<INDEX_TYPE> colrange( 0, cols );

  RAJA::kernel_param<EXEC_POLICY> (
    RAJA::make_tuple( colrange, rowrange ),
    RAJA::make_tuple( RAJA::TileSize{tile_dim_x}, RAJA::TileSize{tile_dim_y} ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr ) {
      WorkTView( cc, rr ) = WorkView( rr, cc );
  });

  work_res.memcpy( check_array_t, work_array_t, sizeof(DATA_TYPE) * array_length );

  for ( int rr = 0; rr < rows; ++rr )
  {
    for ( int cc = 0; cc < cols; ++cc )
    {
      ASSERT_EQ(CheckTView(cc, rr), HostTView(cc, rr));
    }
  }

  // reset check and work transpose arrays
  work_res.memcpy( check_array_t, test_array, sizeof(DATA_TYPE) * array_length );
  work_res.memcpy( work_array_t, test_array, sizeof(DATA_TYPE) * array_length );

  // transpose work_array again with different tile sizes
  RAJA::kernel_param<EXEC_POLICY> (
    RAJA::make_tuple( colrange, rowrange ),
    RAJA::make_tuple( RAJA::TileSize{tile_dim_x}, RAJA::TileSize{tile_dim_y/2} ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr ) {
      WorkTView( cc, rr ) = WorkView( rr, cc );
  });

  work_res.memcpy( check_array_t, work_array_t, sizeof(DATA_TYPE) * array_length );

  for ( int rr = 0; rr < rows; ++rr )
  {
    for ( int cc = 0; cc < cols; ++cc )
    {
      ASSERT_EQ(CheckTView(cc, rr), HostTView(cc, rr));
    }
  }

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array_t,
                                        check_array_t,
                                        test_array_t
                                      );
}


TYPED_TEST_SUITE_P(KernelTileDynamic2DTest);
template <typename T>
class KernelTileDynamic2DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelTileDynamic2DTest, TileDynamic2DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;

  KernelTileDynamic2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(10, 10);
  KernelTileDynamic2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(151, 111);
  KernelTileDynamic2DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileDynamic2DTest,
                            TileDynamic2DKernel);

#endif  // __TEST_KERNEL_TILE_DYNAMIC2D_HPP__
