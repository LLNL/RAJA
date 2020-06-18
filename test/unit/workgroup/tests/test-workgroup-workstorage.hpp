//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_WORKGROUP_WORKSTORAGE__
#define __TEST_WORKGROUP_WORKSTORAGE__

#include "RAJA_test-workgroup.hpp"

#include <random>
#include <array>
#include <cstddef>


template <typename T>
class WorkGroupBasicWorkStorageUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageUnitTest);


template < typename T >
struct TestCallable
{
  TestCallable(T _val)
    : val(_val)
  { }

  TestCallable(TestCallable const&) = delete;
  TestCallable& operator=(TestCallable const&) = delete;

  TestCallable(TestCallable&& o)
    : val(o.val)
    , move_constructed(true)
  {
    o.moved_from = true;
  }

  TestCallable& operator=(TestCallable&& o)
  {
    val = o.val;
    o.moved_from = true;
    return *this;
  }

  RAJA_HOST_DEVICE void operator()(
      void* val_ptr, bool* move_constructed_ptr, bool* moved_from_ptr) const
  {
    *static_cast<T*>(val_ptr) = val;
    *move_constructed_ptr = move_constructed;
    *moved_from_ptr = moved_from;
  }

private:
  T val;
public:
  bool move_constructed = false;
  bool moved_from = false;
};


template <typename StoragePolicy,
          typename Allocator
          >
void testWorkGroupWorkStorageConstructor()
{
  bool success = true;

  using Vtable_type = RAJA::detail::Vtable<void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Vtable_type
                                                    >;

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    WorkStorage_type container2(std::move(container));

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    ASSERT_EQ(container2.size(), (size_t)(0));
    ASSERT_EQ(container2.storage_size(), (size_t)0);
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicWorkStorageUnitTest, BasicWorkGroupWorkStorageConstructor)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageConstructor< StoragePolicy, Allocator >();
}


template <typename StoragePolicy,
          typename Allocator
          >
void testWorkGroupWorkStorageInsert()
{
  bool success = true;

  using Vtable_type = RAJA::detail::Vtable<void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Vtable_type
                                                    >;

  using callable = TestCallable<short>;

  Vtable_type vtable = RAJA::detail::get_Vtable<
      callable, Vtable_type>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    container.template emplace<callable>(&vtable, callable{0});

    ASSERT_EQ(container.size(), (size_t)1);
    ASSERT_TRUE(container.storage_size() >= sizeof(callable));

    WorkStorage_type container2(std::move(container));

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    ASSERT_EQ(container2.size(), (size_t)1);
    ASSERT_TRUE(container2.storage_size() >= sizeof(callable));
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicWorkStorageUnitTest, BasicWorkGroupWorkStorageInsert)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageInsert< StoragePolicy, Allocator >();
}


template <typename StoragePolicy,
          typename Allocator
          >
void testWorkGroupWorkStorageIterator()
{
  bool success = true;

  using Vtable_type = RAJA::detail::Vtable<void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Vtable_type
                                                    >;

  using callable = TestCallable<int>;

  Vtable_type vtable = RAJA::detail::get_Vtable<
      callable, Vtable_type>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.end()-container.begin(), (std::ptrdiff_t)0);
    ASSERT_FALSE(container.begin() < container.end());
    ASSERT_FALSE(container.begin() > container.end());
    ASSERT_TRUE(container.begin() == container.end());
    ASSERT_FALSE(container.begin() != container.end());
    ASSERT_TRUE(container.begin() <= container.end());
    ASSERT_TRUE(container.begin() >= container.end());

    container.template emplace<callable>(&vtable, callable{-1});

    ASSERT_EQ(container.end()-container.begin(), (std::ptrdiff_t)1);
    ASSERT_TRUE(container.begin() < container.end());
    ASSERT_FALSE(container.begin() > container.end());
    ASSERT_FALSE(container.begin() == container.end());
    ASSERT_TRUE(container.begin() != container.end());
    ASSERT_TRUE(container.begin() <= container.end());
    ASSERT_FALSE(container.begin() >= container.end());

    {
      auto iter = container.begin();

      ASSERT_EQ(&*iter, &iter[0]);

      ASSERT_EQ(iter++, container.begin());
      ASSERT_EQ(iter--, container.end());
      ASSERT_EQ(++iter, container.end());
      ASSERT_EQ(--iter, container.begin());

      ASSERT_EQ(iter+1, container.end());
      ASSERT_EQ(1+iter, container.end());
      ASSERT_EQ(++iter, container.end());
      ASSERT_EQ(iter-1, container.begin());
      ASSERT_EQ(iter-=1, container.begin());
      ASSERT_EQ(iter+=1, container.end());
    }
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicWorkStorageUnitTest, BasicWorkGroupWorkStorageIterator)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageIterator< StoragePolicy, Allocator >();
}


template <typename StoragePolicy,
          typename Allocator
          >
void testWorkGroupWorkStorageInsertCall()
{
  bool success = true;

  using Vtable_type = RAJA::detail::Vtable<void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Vtable_type
                                                    >;
  using WorkStruct_type = typename WorkStorage_type::value_type;

  using callable = TestCallable<double>;

  Vtable_type vtable = RAJA::detail::get_Vtable<
      callable, Vtable_type>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    callable c(1.23456789);

    ASSERT_FALSE(c.move_constructed);
    ASSERT_FALSE(c.moved_from);

    container.template emplace<callable>(&vtable, std::move(c));

    ASSERT_FALSE(c.move_constructed);
    ASSERT_TRUE(c.moved_from);

    ASSERT_EQ(container.size(), (size_t)1);
    ASSERT_TRUE(container.storage_size() >= sizeof(callable));

    {
      auto iter = container.begin();

      double val = -1;
      bool move_constructed = false;
      bool moved_from = true;
      WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

      ASSERT_EQ(val, 1.23456789);
      ASSERT_TRUE(move_constructed);
      ASSERT_FALSE(moved_from);
    }

    WorkStorage_type container2(std::move(container));

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    ASSERT_EQ(container2.size(), (size_t)1);
    ASSERT_TRUE(container2.storage_size() >= sizeof(callable));

    {
      auto iter2 = container2.begin();

      double val = -1;
      bool move_constructed = false;
      bool moved_from = true;
      WorkStruct_type::call(&(*iter2), (void*)&val, &move_constructed, &moved_from);

      ASSERT_EQ(val, 1.23456789);
      ASSERT_TRUE(move_constructed);
      ASSERT_FALSE(moved_from);
    }
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicWorkStorageUnitTest, BasicWorkGroupWorkStorageInsertCall)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageInsertCall< StoragePolicy, Allocator >();
}


template <typename StoragePolicy,
          typename Allocator
          >
void testWorkGroupWorkStorageMultiple(
    const size_t num0, const size_t num1, const size_t num2)
{
  bool success = true;

  using Vtable_type = RAJA::detail::Vtable<void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Vtable_type
                                                    >;
  using WorkStruct_type = typename WorkStorage_type::value_type;

  using type0 = double;
  using type1 = std::array<double, 6>;
  using type2 = std::array<double, 14>;

  using callable0 = TestCallable<type0>;
  using callable1 = TestCallable<type1>;
  using callable2 = TestCallable<type2>;

  Vtable_type vtable0 = RAJA::detail::get_Vtable<
      callable0, Vtable_type>(RAJA::seq_work{});
  Vtable_type vtable1 = RAJA::detail::get_Vtable<
      callable1, Vtable_type>(RAJA::seq_work{});
  Vtable_type vtable2 = RAJA::detail::get_Vtable<
      callable2, Vtable_type>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    std::vector<callable0> vec0;
    vec0.reserve(num0);
    for (size_t i = 0; i < num0; ++i) {
      vec0.emplace_back((type0)-i);
      ASSERT_FALSE(vec0[i].move_constructed);
      ASSERT_FALSE(vec0[i].moved_from);
      container.template emplace<callable0>(&vtable0, std::move(vec0[i]));
      ASSERT_FALSE(vec0[i].move_constructed);
      ASSERT_TRUE (vec0[i].moved_from);
    }

    std::vector<callable1> vec1;
    vec1.reserve(num1);
    for (size_t i = 0; i < num1; ++i) {
      vec1.emplace_back(type1(
          { 100.0+i, 110.0+i, 120.0+i, 130.0+i, 140.0+i, 150.0+i }));
      ASSERT_FALSE(vec1[i].move_constructed);
      ASSERT_FALSE(vec1[i].moved_from);
      container.template emplace<callable1>(&vtable1, std::move(vec1[i]));
      ASSERT_FALSE(vec1[i].move_constructed);
      ASSERT_TRUE (vec1[i].moved_from);
    }

    std::vector<callable2> vec2;
    vec2.reserve(num2);
    for (size_t i = 0; i < num2; ++i) {
      vec2.emplace_back(type2(
          { 1000.0+i, 1010.0+i, 1020.0+i, 1030.0+i, 1040.0+i,
            1050.0+i, 1060.0+i, 1070.0+i, 1080.0+i, 1090.0+i,
            1100.0+i, 1110.0+i, 1120.0+i, 1130.0+i }));
      ASSERT_FALSE(vec2[i].move_constructed);
      ASSERT_FALSE(vec2[i].moved_from);
      container.template emplace<callable2>(&vtable2, std::move(vec2[i]));
      ASSERT_FALSE(vec2[i].move_constructed);
      ASSERT_TRUE (vec2[i].moved_from);
    }

    ASSERT_EQ(container.size(), num0+num1+num2);
    ASSERT_GE(container.storage_size(),
        num0*sizeof(callable0) +
        num1*sizeof(callable1) +
        num2*sizeof(callable2));

    WorkStorage_type container2(std::move(container));

    ASSERT_EQ(container.size(), (size_t)0);
    ASSERT_EQ(container.storage_size(), (size_t)0);

    ASSERT_EQ(container2.size(), num0+num1+num2);
    ASSERT_GE(container2.storage_size(),
        num0*sizeof(callable0) +
        num1*sizeof(callable1) +
        num2*sizeof(callable2));

    {
      auto iter = container2.begin();

      for (size_t i = 0; i < num0; ++i) {
        type0 val = 1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        type0 expected = -i;
        ASSERT_EQ(val, expected);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);

        ++iter;
      }

      for (size_t i = 0; i < num1; ++i) {
        type1 val({ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 });
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        type1 expected({
            100.0+i, 110.0+i, 120.0+i, 130.0+i, 140.0+i, 150.0+i });
        ASSERT_EQ(val, expected);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);

        ++iter;
      }

      for (size_t i = 0; i < num2; ++i) {
        type2 val({ -1l, -1l, -1l, -1l, -1l,
                    -1l, -1l, -1l, -1l, -1l,
                    -1l, -1l, -1l, -1l });
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        type2 expected({ 1000.0+i, 1010.0+i, 1020.0+i, 1030.0+i, 1040.0+i,
                         1050.0+i, 1060.0+i, 1070.0+i, 1080.0+i, 1090.0+i,
                         1100.0+i, 1110.0+i, 1120.0+i, 1130.0+i });
        ASSERT_EQ(val, expected);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);

        ++iter;
      }

      ASSERT_EQ(iter, container2.end());
    }
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicWorkStorageUnitTest, BasicWorkGroupWorkStorageMultiple)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupWorkStorageMultiple< StoragePolicy, Allocator >(
      dist(rng), dist(rng), dist(rng));
}


REGISTER_TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageUnitTest,
                            BasicWorkGroupWorkStorageConstructor,
                            BasicWorkGroupWorkStorageInsert,
                            BasicWorkGroupWorkStorageIterator,
                            BasicWorkGroupWorkStorageInsertCall,
                            BasicWorkGroupWorkStorageMultiple);

#endif  //__TEST_WORKGROUP_WORKSTORAGE__
