//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for GPU IndexGlobal
///

//
// test/include headers
//
#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_unit-test-for3d3d.hpp"


template < template < RAJA::named_dim, int, int > class T >
struct indexer_holder
{
  template < RAJA::named_dim dim, int BLOCK_SIZE, int GRID_SIZE >
  using type = T<dim, BLOCK_SIZE, GRID_SIZE>;
};

#if defined(RAJA_ENABLE_HIP)

using HIP_indexer_types =
    camp::list<
                indexer_holder<RAJA::hip::IndexGlobal>
              >;

#endif

using dim_types =
    camp::list<
                camp::integral_constant<RAJA::named_dim, RAJA::named_dim::x>,
                camp::integral_constant<RAJA::named_dim, RAJA::named_dim::y>,
                camp::integral_constant<RAJA::named_dim, RAJA::named_dim::z>
              >;

using size_types =
    camp::list<
                camp::integral_constant<int, RAJA::named_usage::ignored>,
                camp::integral_constant<int, RAJA::named_usage::unspecified>,
                camp::integral_constant<int, 1>,
                camp::integral_constant<int, 7>
              >;




// Pure HIP test.
#if defined(RAJA_ENABLE_HIP)

using HIP_test_types =
  Test< camp::cartesian_product<HipFor3d3dList,
                                HIP_indexer_types,
                                dim_types,
                                size_types,
                                size_types>>::Types;


template <typename T>
class IndexerUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( IndexerUnitTest );

GPU_TYPED_TEST_P( IndexerUnitTest, HIPIndexer )
{
  using for3d3d_policy = typename camp::at<TypeParam, camp::num<0>>::type;
  using indexer_holder_type = typename camp::at<TypeParam, camp::num<1>>::type;
  using dim_type = typename camp::at<TypeParam, camp::num<2>>::type;
  using threads_type = typename camp::at<TypeParam, camp::num<3>>::type;
  using blocks_type = typename camp::at<TypeParam, camp::num<4>>::type;

  using indexer_type = typename indexer_holder_type::template type<
      dim_type::value, threads_type::value, blocks_type::value>;

  dim3d3d expected_dim{{1,1,1}, {1,1,1}};
  if (threads_type::value != RAJA::named_usage::ignored) {
    if (threads_type::value == RAJA::named_usage::unspecified) {
      expected_dim.thread[static_cast<int>(dim_type::value)] = 5;
    } else {
      expected_dim.thread[static_cast<int>(dim_type::value)] = threads_type::value;
    }
  }

  if (blocks_type::value != RAJA::named_usage::ignored) {
    if (blocks_type::value == RAJA::named_usage::unspecified) {
      expected_dim.block[static_cast<int>(dim_type::value)] = 5;
    } else {
      expected_dim.block[static_cast<int>(dim_type::value)] = blocks_type::value;
    }
  }

  int total_global = expected_dim.product();
  int* actual_index = nullptr;
  int* actual_size = nullptr;
  hipErrchk(hipMallocManaged((void **)&actual_index, total_global*sizeof(int)));
  hipErrchk(hipMallocManaged((void **)&actual_size, total_global*sizeof(int)));
  hipErrchk(hipMemset(actual_index, -1, total_global*sizeof(int)));
  hipErrchk(hipMemset(actual_size, -1, total_global*sizeof(int)));

  for3d3d<for3d3d_policy>(expected_dim,
      [=] RAJA_HOST_DEVICE (dim3d3d idx, dim3d3d dim) {
    int i = index(idx, dim);
    actual_index[i] = indexer_type::template index<int>();
    actual_size[i] = indexer_type::template size<int>();
  });
  hipErrchk(hipDeviceSynchronize());

  for (int i = 0; i < total_global; ++i) {
    ASSERT_EQ( actual_index[i], i );
    ASSERT_EQ( actual_size[i], total_global );
  }

  hipErrchk(hipFree(actual_index));
  hipErrchk(hipFree(actual_size));
}

REGISTER_TYPED_TEST_SUITE_P( IndexerUnitTest,
                             HIPIndexer
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( HIPIndexerUnitTest,
                                IndexerUnitTest,
                                HIP_test_types
                              );
#endif

