//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_HYPERPLANE_3D_HPP__
#define __TEST_KERNEL_HYPERPLANE_3D_HPP__

#include <numeric>
#include <type_traits>

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type
KernelHyperplane3DTestImpl(const int RAJA_UNUSED_ARG(groups), const int RAJA_UNUSED_ARG(idim), const int RAJA_UNUSED_ARG(jdim), const int RAJA_UNUSED_ARG(kdim))
{
  // do nothing for unsigned index types
}

template <typename INDEX_TYPE, typename DATA_TYPE, typename WORKING_RES, typename EXEC_POLICY, typename REDUCE_POLICY>
typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type
KernelHyperplane3DTestImpl(const int groups, const int idimin, const int jdimin, const int kdimin)
{
  // This test traverses "groups" number of 3D arrays, and modifies values in a 2D hyperplane manner.

  int idim, jdim, kdim;
  if ( std::is_same<DATA_TYPE, float>::value )
  {
    // Restrict to a small data size for better float precision.
    idim = 5;
    jdim = 5;
    kdim = 5;
  }
  else
  {
    idim = idimin;
    jdim = jdimin;
    kdim = kdimin;
  }

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  DATA_TYPE * work_array;
  DATA_TYPE * check_array;
  DATA_TYPE * test_array;

  INDEX_TYPE array_length = groups * idim * jdim * kdim;

  allocateForallTestData<DATA_TYPE> ( array_length,
                                      work_res,
                                      &work_array,
                                      &check_array,
                                      &test_array
                                    );

  RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>> HostView( test_array, groups, idim, jdim, kdim );
  RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>> WorkView( work_array, groups, idim, jdim, kdim );
  RAJA::View<DATA_TYPE, RAJA::Layout<4, INDEX_TYPE>> CheckView( check_array, groups, idim, jdim, kdim );

  // initialize array
  std::iota( test_array, test_array + array_length, 1 );

  work_res.memcpy( work_array, test_array, sizeof(DATA_TYPE) * array_length );

  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> trip_count(0);
  RAJA::ReduceSum<REDUCE_POLICY, DATA_TYPE> oob_count(0);

  // perform array arithmetic with a 2D J-K hyperplane
  RAJA::TypedRangeSegment<INDEX_TYPE>   Grange( 0, groups );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Irange( 0, idim, 1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Jrange( jdim-1, -1, -1 );
  RAJA::TypedRangeStrideSegment<INDEX_TYPE>  Krange( 0, kdim, 1 );

  RAJA::kernel<EXEC_POLICY> ( RAJA::make_tuple( Grange, Irange, Jrange, Krange ),
    [=] RAJA_HOST_DEVICE ( INDEX_TYPE g, INDEX_TYPE ii, INDEX_TYPE jj, INDEX_TYPE kk ) {
      if (g < 0 || g >= groups || ii < 0 || ii >= idim || jj < 0 || jj >= jdim || kk < 0 || kk >= kdim) {
        oob_count += 1;
      }

      DATA_TYPE left = 1;
      if (ii > 0) {
        left = WorkView(g, ii - 1, jj, kk);
      }

      DATA_TYPE up = 1;
      if (jj > 0) {
        up = WorkView(g, ii, jj - 1, kk);
      }

      DATA_TYPE back = 1;
      if (kk > 0) {
        back = WorkView(g, ii, jj, kk - 1);
      }

      WorkView(g, ii, jj, kk) = left + up + back;

      trip_count += 1;
  });

  work_res.memcpy( check_array, work_array, sizeof(DATA_TYPE) * array_length );

  ASSERT_EQ((INDEX_TYPE)trip_count.get(), (INDEX_TYPE)groups * idim * jdim * kdim);
  ASSERT_EQ((INDEX_TYPE)oob_count.get(), (INDEX_TYPE)0);

  // perform array arithmetic on the CPU
  for (int g = 0; g < groups; ++g) {
    for (int i = 0; i < idim; ++i) {
      for (int j = jdim - 1; j >= 0; --j) {
        for (int k = 0; k < kdim; ++k) {
          DATA_TYPE left = 1;
          if (i > 0) {
            left = HostView(g, i - 1, j, k);
          }

          DATA_TYPE up = 1;
          if (j > 0) {
            up = HostView(g, i, j - 1, k);
          }

          DATA_TYPE back = 1;
          if (k > 0) {
            back = HostView(g, i, j, k - 1);
          }

          HostView(g, i, j, k) = left + up + back;
        }
      }
    }
  }

  for (int g = 0; g < groups; ++g) {
    for (int i = 0; i < idim; ++i) {
      for (int j = 0; j < jdim; ++j) {
        for (int k = 0; k < kdim; ++k) {
          ASSERT_FLOAT_EQ(CheckView(g, i, j, k), HostView(g, i, j, k));
        }
      }
    }
  }

  deallocateForallTestData<DATA_TYPE> ( work_res,
                                        work_array,
                                        check_array,
                                        test_array
                                      );
}


TYPED_TEST_SUITE_P(KernelHyperplane3DTest);
template <typename T>
class KernelHyperplane3DTest : public ::testing::Test
{
};

TYPED_TEST_P(KernelHyperplane3DTest, Hyperplane3DKernel)
{
  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE  = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;

  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(1, 10, 10, 10);
  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(2, 151, 111, 205);
  KernelHyperplane3DTestImpl<INDEX_TYPE, DATA_TYPE, WORKING_RES, EXEC_POLICY, REDUCE_POLICY>(3, 101, 213, 123);
}

REGISTER_TYPED_TEST_SUITE_P(KernelHyperplane3DTest,
                            Hyperplane3DKernel);

#endif  // __TEST_KERNEL_HYPERPLANE_3D_HPP__
