//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup ordered runs.
///

#ifndef __TEST_WORKGROUP_ORDERED_MULTIPLEREUSE__
#define __TEST_WORKGROUP_ORDERED_MULTIPLEREUSE__

#include "RAJA_test-workgroup.hpp"
#include "RAJA_test-forall-data.hpp"

#include <random>
#include <vector>


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
void testWorkGroupOrderedMultiple(
    std::mt19937& rng, IndexType max_begin, IndexType min_end,
    IndexType num1, IndexType num2, IndexType num3,
    IndexType pool_reuse, IndexType group_reuse)
{
  using WorkPool_type = RAJA::WorkPool<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkGroup_type = RAJA::WorkGroup<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkSite_type = RAJA::WorkSite<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  ASSERT_GT(min_end, max_begin);
  IndexType N = min_end + max_begin;

  std::vector<IndexType> begin1, end1;
  std::vector<IndexType> begin2, end2;
  std::vector<IndexType> begin3, end3;

  {
    using dist_type = std::uniform_int_distribution<IndexType>;

    for (IndexType j = IndexType(0); j < num1; j++) {
      begin1.push_back(dist_type(max_begin, min_end-1)(rng));
      end1.push_back(dist_type(begin1.back(), min_end)(rng));
    }

    for (IndexType j = IndexType(0); j < num2; j++) {
      begin2.push_back(dist_type(max_begin, min_end-1)(rng));
      end2.push_back(dist_type(begin2.back(), min_end)(rng));
    }

    for (IndexType j = IndexType(0); j < num3; j++) {
      begin3.push_back(dist_type(max_begin, min_end-1)(rng));
      end3.push_back(dist_type(begin3.back(), min_end)(rng));
    }
  }


  camp::resources::Resource working_res{WORKING_RES::get_default()};

  using type1 = IndexType;
  using type2 = size_t;
  using type3 = double;

  type1* working_array1 = nullptr;
  type1* check_array1 = nullptr;
  type1* test_array1 = nullptr;

  type2* working_array2 = nullptr;
  type2* check_array2 = nullptr;
  type2* test_array2 = nullptr;

  type3* working_array3 = nullptr;
  type3* check_array3 = nullptr;
  type3* test_array3 = nullptr;

  allocateForallTestData<type1>(N * num1,
                                working_res,
                                &working_array1,
                                &check_array1,
                                &test_array1);

  allocateForallTestData<type2>(N * num2,
                                working_res,
                                &working_array2,
                                &check_array2,
                                &test_array2);

  allocateForallTestData<type3>(N * num3,
                                working_res,
                                &working_array3,
                                &check_array3,
                                &test_array3);


  WorkPool_type pool(Allocator{});
  WorkGroup_type group = pool.instantiate();
  WorkSite_type site = group.run();

  for (IndexType pr = 0; pr < pool_reuse; pr++) {

    type1 test_val1(5);
    type2 test_val2(7);
    type3 test_val3(11);

    // fill_pool(pool, type1(5), type2(7), type3(11));
    {
      for (IndexType j = IndexType(0); j < num1; j++) {
        type1* working_ptr1 = working_array1 + N * j;
        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin1[j], end1[j] },
            [=] RAJA_HOST_DEVICE (IndexType i) {
          working_ptr1[i] += type1(i);
        });
        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin1[j], end1[j] },
            [=] RAJA_HOST_DEVICE (IndexType i) {
          working_ptr1[i] += test_val1;
        });
      }

      for (IndexType j = IndexType(0); j < num2; j++) {
        type2* working_ptr2 = working_array2 + N * j;
        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin2[j], end2[j] },
            [=] RAJA_HOST_DEVICE (IndexType i) {
          working_ptr2[i] += type2(i);
        });
        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin2[j], end2[j] },
            [=] RAJA_HOST_DEVICE (IndexType i) {
          working_ptr2[i] += test_val2;
        });
      }

      for (IndexType j = IndexType(0); j < num3; j++) {
        type3* working_ptr3 = working_array3 + N * j;
        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin3[j], end3[j] },
            [=] RAJA_HOST_DEVICE (IndexType i) {
          working_ptr3[i] += type3(i);
        });
        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin3[j], end3[j] },
            [=] RAJA_HOST_DEVICE (IndexType i) {
          working_ptr3[i] += test_val3;
        });
      }
    }

    group = pool.instantiate();

    for (IndexType gr = 0; gr < group_reuse; gr++) {

      // set_test_data();
      {
        for (IndexType j = IndexType(0); j < num1; j++) {
          type1* test_ptr1 = test_array1 + N * j;
          for (IndexType i = IndexType(0); i < N; i++) {
            test_ptr1[i] = type1(0);
          }
        }

        for (IndexType j = IndexType(0); j < num2; j++) {
          type2* test_ptr2 = test_array2 + N * j;
          for (IndexType i = IndexType(0); i < N; i++) {
            test_ptr2[i] = type2(0);
          }
        }

        for (IndexType j = IndexType(0); j < num3; j++) {
          type3* test_ptr3 = test_array3 + N * j;
          for (IndexType i = IndexType(0); i < N; i++) {
            test_ptr3[i] = type3(0);
          }
        }


        working_res.memcpy(working_array1, test_array1, sizeof(type1) * N * num1);

        working_res.memcpy(working_array2, test_array2, sizeof(type2) * N * num2);

        working_res.memcpy(working_array3, test_array3, sizeof(type3) * N * num3);


        for (IndexType j = IndexType(0); j < num1; j++) {
          type1* test_ptr1 = test_array1 + N * j;
          for (IndexType i = begin1[j]; i < end1[j]; ++i) {
            test_ptr1[ i ] = type1(i);
          }
        }

        for (IndexType j = IndexType(0); j < num2; j++) {
          type2* test_ptr2 = test_array2 + N * j;
          for (IndexType i = begin2[j]; i < end2[j]; ++i) {
            test_ptr2[ i ] = type2(i);
          }
        }

        for (IndexType j = IndexType(0); j < num3; j++) {
          type3* test_ptr3 = test_array3 + N * j;
          for (IndexType i = begin3[j]; i < end3[j]; ++i) {
            test_ptr3[ i ] = type3(i);
          }
        }
      }

      site = group.run();

      // check_test_data(type1(5), type2(7), type3(11));
      {
        working_res.memcpy(check_array1, working_array1, sizeof(type1) * N * num1);

        working_res.memcpy(check_array2, working_array2, sizeof(type2) * N * num2);

        working_res.memcpy(check_array3, working_array3, sizeof(type3) * N * num3);


        for (IndexType j = IndexType(0); j < num1; j++) {
          type1* test_ptr1 = test_array1 + N * j;
          type1* check_ptr1 = check_array1 + N * j;
          for (IndexType i = IndexType(0); i < begin1[j]; i++) {
            ASSERT_EQ(test_ptr1[i], check_ptr1[i]);
          }
          for (IndexType i = begin1[j];    i < end1[j];   i++) {
            ASSERT_EQ(test_ptr1[i] + test_val1, check_ptr1[i]);
          }
          for (IndexType i = end1[j];      i < N;     i++) {
            ASSERT_EQ(test_ptr1[i], check_ptr1[i]);
          }
        }

        for (IndexType j = IndexType(0); j < num2; j++) {
          type2* test_ptr2 = test_array2 + N * j;
          type2* check_ptr2 = check_array2 + N * j;
          for (IndexType i = IndexType(0); i < begin2[j]; i++) {
            ASSERT_EQ(test_ptr2[i], check_ptr2[i]);
          }
          for (IndexType i = begin2[j];    i < end2[j];   i++) {
            ASSERT_EQ(test_ptr2[i] + test_val2, check_ptr2[i]);
          }
          for (IndexType i = end2[j];      i < N;     i++) {
            ASSERT_EQ(test_ptr2[i], check_ptr2[i]);
          }
        }

        for (IndexType j = IndexType(0); j < num3; j++) {
          type3* test_ptr3 = test_array3 + N * j;
          type3* check_ptr3 = check_array3 + N * j;
          for (IndexType i = IndexType(0); i < begin3[j]; i++) {
            ASSERT_EQ(test_ptr3[i], check_ptr3[i]);
          }
          for (IndexType i = begin3[j];    i < end3[j];   i++) {
            ASSERT_EQ(test_ptr3[i] + test_val3, check_ptr3[i]);
          }
          for (IndexType i = end3[j];      i < N;     i++) {
            ASSERT_EQ(test_ptr3[i], check_ptr3[i]);
          }
        }
      }
    }

    site.clear();
    group.clear();
    pool.clear();
  }


  deallocateForallTestData<type1>(working_res,
                                  working_array1,
                                  check_array1,
                                  test_array1);

  deallocateForallTestData<type2>(working_res,
                                  working_array2,
                                  check_array2,
                                  test_array2);

  deallocateForallTestData<type3>(working_res,
                                  working_array3,
                                  check_array3,
                                  test_array3);
}


template <typename T>
class WorkGroupBasicOrderedMultipleReuseFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicOrderedMultipleReuseFunctionalTest);


TYPED_TEST_P(WorkGroupBasicOrderedMultipleReuseFunctionalTest, BasicWorkGroupOrderedMultipleReuse)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType num1 = dist_type(IndexType(0), IndexType(8))(rng);
  IndexType num2 = dist_type(IndexType(0), IndexType(8))(rng);
  IndexType num3 = dist_type(IndexType(0), IndexType(8))(rng);

  IndexType pool_reuse  = dist_type(IndexType(0), IndexType(8))(rng);
  IndexType group_reuse = dist_type(IndexType(0), IndexType(8))(rng);

  testWorkGroupOrderedMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(
      rng, IndexType(96), IndexType(4000), num1, num2, num3, pool_reuse, group_reuse);
}

#endif  //__TEST_WORKGROUP_ORDERED_MULTIPLEREUSE__
