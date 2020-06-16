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

#include "../test-workgroup.hpp"

#include <vector>
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

  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      void*, bool*, bool*
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

  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      void*, bool*, bool*
                                                    >;
  using Vtable_type = typename WorkStorage_type::vtable_type;

  Vtable_type vtable = RAJA::detail::get_Vtable<
      TestCallable<short>, void*, bool*, bool*>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    container.insert(&vtable, TestCallable<short>{0});

    ASSERT_EQ(container.size(), (size_t)1);
    ASSERT_TRUE(container.storage_size() >= sizeof(TestCallable<short>));

    WorkStorage_type container2(std::move(container));

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    ASSERT_EQ(container2.size(), (size_t)1);
    ASSERT_TRUE(container2.storage_size() >= sizeof(TestCallable<short>));
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

  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      void*, bool*, bool*
                                                    >;
  using Vtable_type = typename WorkStorage_type::vtable_type;

  Vtable_type vtable = RAJA::detail::get_Vtable<
      TestCallable<int>, void*, bool*, bool*>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.end()-container.begin(), (std::ptrdiff_t)0);
    ASSERT_FALSE(container.begin() < container.end());
    ASSERT_FALSE(container.begin() > container.end());
    ASSERT_TRUE(container.begin() == container.end());
    ASSERT_FALSE(container.begin() != container.end());
    ASSERT_TRUE(container.begin() <= container.end());
    ASSERT_TRUE(container.begin() >= container.end());

    container.insert(&vtable, TestCallable<int>{-1});

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

  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      void*, bool*, bool*
                                                    >;
  using WorkStruct_type = typename WorkStorage_type::value_type;
  using Vtable_type = typename WorkStorage_type::vtable_type;

  Vtable_type vtable = RAJA::detail::get_Vtable<
      TestCallable<double>, void*, bool*, bool*>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    TestCallable<double> c(1.23456789);

    ASSERT_FALSE(c.move_constructed);
    ASSERT_FALSE(c.moved_from);

    container.insert(&vtable, std::move(c));

    ASSERT_FALSE(c.move_constructed);
    ASSERT_TRUE(c.moved_from);

    ASSERT_EQ(container.size(), (size_t)1);
    ASSERT_TRUE(container.storage_size() >= sizeof(TestCallable<double>));

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
    ASSERT_TRUE(container2.storage_size() >= sizeof(TestCallable<double>));

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
void testWorkGroupWorkStorageMultiple()
{
  bool success = true;

  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      void*, bool*, bool*
                                                    >;
  using WorkStruct_type = typename WorkStorage_type::value_type;
  using Vtable_type = typename WorkStorage_type::vtable_type;

  Vtable_type vtable0 = RAJA::detail::get_Vtable<
      TestCallable<double>, void*, bool*, bool*>(RAJA::seq_work{});
  Vtable_type vtable1 = RAJA::detail::get_Vtable<
      TestCallable<float>, void*, bool*, bool*>(RAJA::seq_work{});
  Vtable_type vtable2 = RAJA::detail::get_Vtable<
      TestCallable<std::vector<int>>, void*, bool*, bool*>(RAJA::seq_work{});

  {
    WorkStorage_type container(Allocator{});

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    TestCallable<double> c01(0.12345);
    TestCallable<double> c02(0.23451);
    TestCallable<double> c03(0.34512);
    TestCallable<double> c04(0.45123);
    TestCallable<double> c05(0.51234);

    TestCallable<float> c11(1.1234f);
    TestCallable<float> c12(1.2341f);
    TestCallable<float> c13(1.3412f);
    TestCallable<float> c14(1.4123f);

    TestCallable<std::vector<int>> c21({2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    ASSERT_FALSE(c01.move_constructed);
    ASSERT_FALSE(c01.moved_from);
    ASSERT_FALSE(c02.move_constructed);
    ASSERT_FALSE(c02.moved_from);
    ASSERT_FALSE(c03.move_constructed);
    ASSERT_FALSE(c03.moved_from);
    ASSERT_FALSE(c04.move_constructed);
    ASSERT_FALSE(c04.moved_from);
    ASSERT_FALSE(c05.move_constructed);
    ASSERT_FALSE(c05.moved_from);

    ASSERT_FALSE(c11.move_constructed);
    ASSERT_FALSE(c11.moved_from);
    ASSERT_FALSE(c12.move_constructed);
    ASSERT_FALSE(c12.moved_from);
    ASSERT_FALSE(c13.move_constructed);
    ASSERT_FALSE(c13.moved_from);
    ASSERT_FALSE(c14.move_constructed);
    ASSERT_FALSE(c14.moved_from);

    ASSERT_FALSE(c21.move_constructed);
    ASSERT_FALSE(c21.moved_from);

    container.insert(&vtable0, std::move(c01));
    container.insert(&vtable0, std::move(c02));
    container.insert(&vtable0, std::move(c03));
    container.insert(&vtable0, std::move(c04));
    container.insert(&vtable0, std::move(c05));

    container.insert(&vtable1, std::move(c11));
    container.insert(&vtable1, std::move(c12));
    container.insert(&vtable1, std::move(c13));
    container.insert(&vtable1, std::move(c14));

    container.insert(&vtable2, std::move(c21));

    ASSERT_FALSE(c01.move_constructed);
    ASSERT_TRUE(c01.moved_from);
    ASSERT_FALSE(c02.move_constructed);
    ASSERT_TRUE(c02.moved_from);
    ASSERT_FALSE(c03.move_constructed);
    ASSERT_TRUE(c03.moved_from);
    ASSERT_FALSE(c04.move_constructed);
    ASSERT_TRUE(c04.moved_from);
    ASSERT_FALSE(c05.move_constructed);
    ASSERT_TRUE(c05.moved_from);

    ASSERT_FALSE(c11.move_constructed);
    ASSERT_TRUE(c11.moved_from);
    ASSERT_FALSE(c12.move_constructed);
    ASSERT_TRUE(c12.moved_from);
    ASSERT_FALSE(c13.move_constructed);
    ASSERT_TRUE(c13.moved_from);
    ASSERT_FALSE(c14.move_constructed);
    ASSERT_TRUE(c14.moved_from);

    ASSERT_FALSE(c21.move_constructed);
    ASSERT_TRUE(c21.moved_from);

    ASSERT_EQ(container.size(), (size_t)10);
    ASSERT_GE(container.storage_size(),
        5*sizeof(TestCallable<double>) +
        4*sizeof(TestCallable<float>) +
        4*sizeof(TestCallable<std::vector<int>>));

    WorkStorage_type container2(std::move(container));

    ASSERT_EQ(container.size(), (size_t)(0));
    ASSERT_EQ(container.storage_size(), (size_t)0);

    ASSERT_EQ(container2.size(), (size_t)10);
    ASSERT_GE(container2.storage_size(),
        5*sizeof(TestCallable<double>) +
        4*sizeof(TestCallable<float>) +
        4*sizeof(TestCallable<std::vector<int>>));

    {
      auto iter = container2.begin();

      {
        double val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 0.12345);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        double val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 0.23451);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        double val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 0.34512);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        double val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 0.45123);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        double val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 0.51234);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;


      {
        float val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 1.1234f);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        float val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 1.2341f);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        float val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 1.3412f);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      {
        float val = -1;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        ASSERT_EQ(val, 1.4123f);
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;


      {
        std::vector<int> val;
        bool move_constructed = false;
        bool moved_from = true;
        WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

        for (int i = 2; i < 12; ++i) {
          ASSERT_EQ(val[i-2], i);
        }
        ASSERT_TRUE(move_constructed);
        ASSERT_FALSE(moved_from);
      }

      ++iter;

      ASSERT_EQ(iter, container2.end());
    }
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicWorkStorageUnitTest, BasicWorkGroupWorkStorageMultiple)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageMultiple< StoragePolicy, Allocator >();
}


REGISTER_TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageUnitTest,
                            BasicWorkGroupWorkStorageConstructor,
                            BasicWorkGroupWorkStorageInsert,
                            BasicWorkGroupWorkStorageIterator,
                            BasicWorkGroupWorkStorageInsertCall,
                            BasicWorkGroupWorkStorageMultiple);

#endif  //__TEST_WORKGROUP_WORKSTORAGE__
