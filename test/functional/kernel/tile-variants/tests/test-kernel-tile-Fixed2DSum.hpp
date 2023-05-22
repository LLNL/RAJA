//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_TILE_FIXED2DSUM_HPP__
#define __TEST_KERNEL_TILE_FIXED2DSUM_HPP__

#include <numeric>
#include <vector>
#include <type_traits>

// Remove testing of float.
// Does not work for matrices of 33x33 or greater, when using tile_size != 2.
template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
typename std::enable_if<
           (std::is_same<DATA_TYPE,float>::value)
         >::type
KernelTileFixed2DSumTestImpl(const int rows, const int cols)
{
  // do nothing for float type
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
typename std::enable_if<
           (!std::is_same<DATA_TYPE,float>::value)
         >::type
KernelTileFixed2DSumTestImpl(const int rows, const int cols)
{
  // This test reduces sums with tiling.

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE hostsum = 0;

  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> worksum( DATA_TYPE(0) ); 

  // sum on CPU
  for ( int rr = 0; rr < rows; ++rr )
  {
    for ( int cc = 0; cc < cols; ++cc )
    {
      hostsum += (DATA_TYPE)(rr * 1.1 + cc);
    }
  }

  // mixed range types
  RAJA::TypedRangeSegment<INDEX_TYPE> rowrange( 0, rows );

  std::vector<INDEX_TYPE> colidx;
  for (INDEX_TYPE ii = INDEX_TYPE(0); ii < cols; ++ii)
  {
    colidx.push_back(ii);
  }

  RAJA::TypedListSegment<INDEX_TYPE> colrange( &colidx[0], colidx.size(), work_res );

  // sum on target platform
  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( colrange, rowrange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE cc, INDEX_TYPE rr ) {
      worksum += (DATA_TYPE)(rr * 1.1 + cc);
  });

  ASSERT_FLOAT_EQ(hostsum, (DATA_TYPE)worksum.get());
}


TYPED_TEST_SUITE_P(KernelTileFixed2DSumTest);
template <typename T>
class KernelTileFixed2DSumTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelTileFixed2DSumTest, TileFixed2DSumKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(10, 10);
  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(151, 111);
  KernelTileFixed2DSumTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(362, 362);
}

REGISTER_TYPED_TEST_SUITE_P(KernelTileFixed2DSumTest,
                            TileFixed2DSumKernel);

#endif  // __TEST_KERNEL_TILE_FIXED2DSUM_HPP__
