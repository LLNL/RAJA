//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup unordered runs.
///

#ifndef __TEST_WORKGROUP_UNORDERED__
#define __TEST_WORKGROUP_UNORDERED__

#include "RAJA_test-workgroup.hpp"
#include "RAJA_test-forall-data.hpp"

#include <random>


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
void testWorkGroupUnorderedSingle(IndexType begin, IndexType end)
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

  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  camp::resources::Resource working_res{WORKING_RES()};

  IndexType* working_array;
  IndexType* check_array;
  IndexType* test_array;

  allocateForallTestData<IndexType>(N,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  auto set_test_data = [&]() {

    for (IndexType i = IndexType(0); i < N; i++) {
      test_array[i] = IndexType(0);
    }

    working_res.memcpy(working_array, test_array, sizeof(IndexType) * N);

    for (IndexType i = begin; i < end; ++i) {
      test_array[ i ] = IndexType(i);
    }
  };

  auto fill_pool = [&](WorkPool_type& pool, IndexType test_val) {

    pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
        [=] RAJA_HOST_DEVICE (IndexType i) {
      working_array[i] += i + test_val;
    });
  };

  auto check_test_data = [&](IndexType test_val) {

    working_res.memcpy(check_array, working_array, sizeof(IndexType) * N);

    for (IndexType i = IndexType(0); i < begin; i++) {
      ASSERT_EQ(test_array[i], check_array[i]);
    }
    for (IndexType i = begin;        i < end;   i++) {
      ASSERT_EQ(test_array[i] + test_val, check_array[i]);
    }
    for (IndexType i = end;          i < N;     i++) {
      ASSERT_EQ(test_array[i], check_array[i]);
    }
  };


  set_test_data();

  WorkPool_type pool(Allocator{});

  fill_pool(pool, IndexType(5));

  WorkGroup_type group = pool.instantiate();

  WorkSite_type site = group.run();

  working_res.memcpy(check_array, working_array, sizeof(IndexType) * N);

  check_test_data(IndexType(5));


  deallocateForallTestData<IndexType>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
void testWorkGroupUnorderedMultiple(
    IndexType begin, IndexType end,
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

  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  camp::resources::Resource working_res{WORKING_RES()};

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


  auto set_test_data = [&]() {

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
      for (IndexType i = begin; i < end; ++i) {
        test_ptr1[ i ] = type1(i);
      }
    }

    for (IndexType j = IndexType(0); j < num2; j++) {
      type2* test_ptr2 = test_array2 + N * j;
      for (IndexType i = begin; i < end; ++i) {
        test_ptr2[ i ] = type2(i);
      }
    }

    for (IndexType j = IndexType(0); j < num3; j++) {
      type3* test_ptr3 = test_array3 + N * j;
      for (IndexType i = begin; i < end; ++i) {
        test_ptr3[ i ] = type3(i);
      }
    }
  };

  auto fill_pool = [&](WorkPool_type& pool, type1 test_val1, type2 test_val2, type3 test_val3) {

    for (IndexType j = IndexType(0); j < num1; j++) {
      type1* working_ptr1 = working_array1 + N * j;
      pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
          [=] RAJA_HOST_DEVICE (IndexType i) {
        working_ptr1[i] += type1(i) + test_val1;
      });
    }

    for (IndexType j = IndexType(0); j < num2; j++) {
      type2* working_ptr2 = working_array2 + N * j;
      pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
          [=] RAJA_HOST_DEVICE (IndexType i) {
        working_ptr2[i] += type2(i) + test_val2;
      });
    }

    for (IndexType j = IndexType(0); j < num3; j++) {
      type3* working_ptr3 = working_array3 + N * j;
      pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
          [=] RAJA_HOST_DEVICE (IndexType i) {
        working_ptr3[i] += type3(i) + test_val3;
      });
    }
  };

  auto check_test_data = [&](type1 test_val1, type2 test_val2, type3 test_val3) {

    working_res.memcpy(check_array1, working_array1, sizeof(type1) * N * num1);

    working_res.memcpy(check_array2, working_array2, sizeof(type2) * N * num2);

    working_res.memcpy(check_array3, working_array3, sizeof(type3) * N * num3);


    for (IndexType j = IndexType(0); j < num1; j++) {
      type1* test_ptr1 = test_array1 + N * j;
      type1* check_ptr1 = check_array1 + N * j;
      for (IndexType i = IndexType(0); i < begin; i++) {
        ASSERT_EQ(test_ptr1[i], check_ptr1[i]);
      }
      for (IndexType i = begin;        i < end;   i++) {
        ASSERT_EQ(test_ptr1[i] + test_val1, check_ptr1[i]);
      }
      for (IndexType i = end;          i < N;     i++) {
        ASSERT_EQ(test_ptr1[i], check_ptr1[i]);
      }
    }

    for (IndexType j = IndexType(0); j < num2; j++) {
      type2* test_ptr2 = test_array2 + N * j;
      type2* check_ptr2 = check_array2 + N * j;
      for (IndexType i = IndexType(0); i < begin; i++) {
        ASSERT_EQ(test_ptr2[i], check_ptr2[i]);
      }
      for (IndexType i = begin;        i < end;   i++) {
        ASSERT_EQ(test_ptr2[i] + test_val2, check_ptr2[i]);
      }
      for (IndexType i = end;          i < N;     i++) {
        ASSERT_EQ(test_ptr2[i], check_ptr2[i]);
      }
    }

    for (IndexType j = IndexType(0); j < num3; j++) {
      type3* test_ptr3 = test_array3 + N * j;
      type3* check_ptr3 = check_array3 + N * j;
      for (IndexType i = IndexType(0); i < begin; i++) {
        ASSERT_EQ(test_ptr3[i], check_ptr3[i]);
      }
      for (IndexType i = begin;        i < end;   i++) {
        ASSERT_EQ(test_ptr3[i] + test_val3, check_ptr3[i]);
      }
      for (IndexType i = end;          i < N;     i++) {
        ASSERT_EQ(test_ptr3[i], check_ptr3[i]);
      }
    }
  };


  WorkPool_type pool(Allocator{});

  for (IndexType pr = 0; pr < pool_reuse; pr++) {

    fill_pool(pool, type1(5), type2(7), type3(11));

    WorkGroup_type group = pool.instantiate();

    for (IndexType gr = 0; gr < group_reuse; gr++) {

      set_test_data();

      WorkSite_type site = group.run();

      check_test_data(type1(5), type2(7), type3(11));
    }
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
class WorkGroupBasicUnorderedSingleFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedSingleFunctionalTest);

template <typename T>
class WorkGroupBasicUnorderedMultipleFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedMultipleFunctionalTest);

template <typename T>
class WorkGroupBasicUnorderedMultipleReusePoolFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedMultipleReusePoolFunctionalTest);

template <typename T>
class WorkGroupBasicUnorderedMultipleReuseGroupFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedMultipleReuseGroupFunctionalTest);

template <typename T>
class WorkGroupBasicUnorderedMultipleReusePoolGroupFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedMultipleReusePoolGroupFunctionalTest);


TYPED_TEST_P(WorkGroupBasicUnorderedSingleFunctionalTest, BasicWorkGroupUnorderedSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType b1 = dist_type(IndexType(0), IndexType(15))(rng);
  IndexType e1 = dist_type(b1, IndexType(16))(rng);

  IndexType b2 = dist_type(e1, IndexType(127))(rng);
  IndexType e2 = dist_type(b2, IndexType(128))(rng);

  IndexType b3 = dist_type(e2, IndexType(1023))(rng);
  IndexType e3 = dist_type(b3, IndexType(1024))(rng);

  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b1, e1);
  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b2, e2);
  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b3, e3);
}

TYPED_TEST_P(WorkGroupBasicUnorderedMultipleFunctionalTest, BasicWorkGroupUnorderedMultiple)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType begin = dist_type(IndexType(1), IndexType(16383))(rng);
  IndexType end   = dist_type(begin,        IndexType(16384))(rng);

  IndexType num1 = dist_type(IndexType(0), IndexType(32))(rng);
  IndexType num2 = dist_type(IndexType(0), IndexType(32))(rng);
  IndexType num3 = dist_type(IndexType(0), IndexType(32))(rng);

  testWorkGroupUnorderedMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(
      begin, end, num1, num2, num3, IndexType(1), IndexType(1));
}

TYPED_TEST_P(WorkGroupBasicUnorderedMultipleReusePoolFunctionalTest, BasicWorkGroupUnorderedMultipleReusePool)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType begin = dist_type(IndexType(1), IndexType(8191))(rng);
  IndexType end   = dist_type(begin,        IndexType(8192))(rng);

  IndexType num1 = dist_type(IndexType(0), IndexType(16))(rng);
  IndexType num2 = dist_type(IndexType(0), IndexType(16))(rng);
  IndexType num3 = dist_type(IndexType(0), IndexType(16))(rng);

  IndexType pool_reuse = dist_type(IndexType(0), IndexType(8))(rng);

  testWorkGroupUnorderedMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(
      begin, end, num1, num2, num3, pool_reuse, IndexType(1));
}

TYPED_TEST_P(WorkGroupBasicUnorderedMultipleReuseGroupFunctionalTest, BasicWorkGroupUnorderedMultipleReuseGroup)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType begin = dist_type(IndexType(1), IndexType(8191))(rng);
  IndexType end   = dist_type(begin,        IndexType(8192))(rng);

  IndexType num1 = dist_type(IndexType(0), IndexType(16))(rng);
  IndexType num2 = dist_type(IndexType(0), IndexType(16))(rng);
  IndexType num3 = dist_type(IndexType(0), IndexType(16))(rng);

  IndexType group_reuse = dist_type(IndexType(0), IndexType(8))(rng);

  testWorkGroupUnorderedMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(
      begin, end, num1, num2, num3, IndexType(1), group_reuse);
}

TYPED_TEST_P(WorkGroupBasicUnorderedMultipleReusePoolGroupFunctionalTest, BasicWorkGroupUnorderedMultipleReusePoolGroup)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType begin = dist_type(IndexType(1), IndexType(4095))(rng);
  IndexType end   = dist_type(begin,        IndexType(4096))(rng);

  IndexType num1 = dist_type(IndexType(0), IndexType(8))(rng);
  IndexType num2 = dist_type(IndexType(0), IndexType(8))(rng);
  IndexType num3 = dist_type(IndexType(0), IndexType(8))(rng);

  IndexType pool_reuse  = dist_type(IndexType(0), IndexType(8))(rng);
  IndexType group_reuse = dist_type(IndexType(0), IndexType(8))(rng);

  testWorkGroupUnorderedMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(
      begin, end, num1, num2, num3, pool_reuse, group_reuse);
}

#endif  //__TEST_WORKGROUP_UNORDERED__
