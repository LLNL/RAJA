//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_SINGLE_LOOP_TILETCOUNT_HPP_
#define __TEST_KERNEL_SINGLE_LOOP_TILETCOUNT_HPP_

//
// Value struct for manipulating tile sizes in parameterized tests.
//
template<int VALUE>
struct Value {
  static constexpr int value = VALUE;
};


template <typename IDX_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY>
void KernelSingleLoopTileTCountTestImpl(IDX_TYPE N, IDX_TYPE tsize)
{

  IDX_TYPE NT = (N + tsize - 1) / tsize;

  RAJA::ReduceSum<REDUCE_POLICY, IDX_TYPE> trip_count(0);

  for (IDX_TYPE t = 0; t < NT; ++t) {

    RAJA::ReduceSum<REDUCE_POLICY, IDX_TYPE> tile_count(0);

    RAJA::kernel_param<EXEC_POLICY>(
      RAJA::make_tuple( RAJA::TypedRangeSegment<IDX_TYPE>(0, N) ),
      RAJA::make_tuple( static_cast<IDX_TYPE>(0) ),

      [=] RAJA_HOST_DEVICE(IDX_TYPE i, IDX_TYPE ti) {
        trip_count += 1;
        if ( i / tsize == t && ti == t ) {
          tile_count += 1;
        }
      }
    );

    IDX_TYPE trip_result = trip_count.get();
    ASSERT_EQ( trip_result, (t+1) * N );

    IDX_TYPE tile_result = tile_count.get();

    IDX_TYPE tile_expect = tsize;
    if ( (t + 1) * tsize > N ) {
      tile_expect = N - t * tsize;
    }
    ASSERT_EQ(tile_result, tile_expect);

  }

}


TYPED_TEST_SUITE_P(KernelSingleLoopTileTCountTest);
template <typename T>
class KernelSingleLoopTileTCountTest : public ::testing::Test
{
};


TYPED_TEST_P(KernelSingleLoopTileTCountTest, TileTCountSingleLoopKernel)
{
  using IDX_TYPE    = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<1>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  IDX_TYPE tsize = camp::at_v<TypeParam, 3>::value;

  KernelSingleLoopTileTCountTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY>(
    IDX_TYPE(57), tsize);
  KernelSingleLoopTileTCountTestImpl<IDX_TYPE, EXEC_POLICY, REDUCE_POLICY>(
    IDX_TYPE(1035), tsize);

}

REGISTER_TYPED_TEST_SUITE_P(KernelSingleLoopTileTCountTest,
                            TileTCountSingleLoopKernel);

#endif  // __TEST_KERNEL_SINGLE_LOOP_TILETCOUNT_HPP_
