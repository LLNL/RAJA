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
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    WorkStorage_type container(Allocator{});

    test_empty(container);

    container.clear();

    test_empty(container);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_empty(container2);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_empty(container3);
  }

  ASSERT_TRUE(success);
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

  const Vtable_type* vtable = RAJA::detail::get_Vtable<
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

    container.template emplace<callable>(vtable, callable{-1});

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

  const Vtable_type* vtable = RAJA::detail::get_Vtable<
      callable, Vtable_type>(RAJA::seq_work{});

  {
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    auto fill_contents = [&](WorkStorage_type& container, double init_val) {

      callable c(init_val);

      ASSERT_FALSE(c.move_constructed);
      ASSERT_FALSE(c.moved_from);

      container.template emplace<callable>(vtable, std::move(c));

      ASSERT_FALSE(c.move_constructed);
      ASSERT_TRUE(c.moved_from);

      ASSERT_EQ(container.size(), (size_t)1);
      ASSERT_TRUE(container.storage_size() >= sizeof(callable));
    };

    auto test_contents = [&](WorkStorage_type& container, double init_val) {

      ASSERT_EQ(container.size(), (size_t)1);
      ASSERT_TRUE(container.storage_size() >= sizeof(callable));

      auto iter = container.begin();

      double test_val = -1;
      bool move_constructed = false;
      bool moved_from = true;
      WorkStruct_type::call(&*iter, (void*)&test_val, &move_constructed, &moved_from);

      ASSERT_EQ(test_val, init_val);
      ASSERT_TRUE(move_constructed);
      ASSERT_FALSE(moved_from);
    };


    WorkStorage_type container(Allocator{});

    test_empty(container);

    container.clear();

    test_empty(container);
    fill_contents(container, 1.23456789);
    test_contents(container, 1.23456789);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_contents(container2, 1.23456789);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_contents(container3, 1.23456789);


    WorkStorage_type container4(Allocator{});

    fill_contents(container4, 2.34567891);
    test_contents(container4, 2.34567891);

    container4 = std::move(container3);

    test_empty(container3);
    test_contents(container4, 1.23456789);
  }

  ASSERT_TRUE(success);
}

// work around inconsistent std::array support over stl versions
template < typename T, size_t N >
struct TestArray
{
  T a[N]{};
  T& operator[](size_t i) { return a[i]; }
  T const& operator[](size_t i) const { return a[i]; }
  friend inline bool operator==(TestArray const& lhs, TestArray const& rhs)
  {
    for (size_t i = 0; i < N; ++i) {
      if (lhs[i] == rhs[i]) continue;
      else return false;
    }
    return true;
  }
  friend inline bool operator!=(TestArray const& lhs, TestArray const& rhs)
  {
    return !(lhs == rhs);
  }
};

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
  using type1 = TestArray<double, 6>;
  using type2 = TestArray<double, 14>;

  auto make_type0 = [](double init_val, size_t i) {
    type0 obj(init_val - (double)i);
    return obj;
  };
  auto make_type1 = [](double init_val, size_t i) {
    type1 obj{};
    for (size_t j = 0; j < 6; ++j) {
      obj[j] = init_val + 10.0 * j + i;
    }
    return obj;
  };
  auto make_type2 = [](double init_val, size_t i) {
    type2 obj{};
    for (size_t j = 0; j < 14; ++j) {
      obj[j] = init_val + 10.0 * j + i;
    }
    return obj;
  };

  using callable0 = TestCallable<type0>;
  using callable1 = TestCallable<type1>;
  using callable2 = TestCallable<type2>;

  const Vtable_type* vtable0 = RAJA::detail::get_Vtable<
      callable0, Vtable_type>(RAJA::seq_work{});
  const Vtable_type* vtable1 = RAJA::detail::get_Vtable<
      callable1, Vtable_type>(RAJA::seq_work{});
  const Vtable_type* vtable2 = RAJA::detail::get_Vtable<
      callable2, Vtable_type>(RAJA::seq_work{});

  {
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    auto fill_contents = [&](WorkStorage_type& container, double init_val0, double init_val1, double init_val2) {

      std::vector<callable0> vec0;
      vec0.reserve(num0);
      for (size_t i = 0; i < num0; ++i) {
        vec0.emplace_back(make_type0(init_val0, i));
        ASSERT_FALSE(vec0[i].move_constructed);
        ASSERT_FALSE(vec0[i].moved_from);
        container.template emplace<callable0>(vtable0, std::move(vec0[i]));
        ASSERT_FALSE(vec0[i].move_constructed);
        ASSERT_TRUE (vec0[i].moved_from);
      }

      std::vector<callable1> vec1;
      vec1.reserve(num1);
      for (size_t i = 0; i < num1; ++i) {
        vec1.emplace_back(make_type1(init_val1, i));
        ASSERT_FALSE(vec1[i].move_constructed);
        ASSERT_FALSE(vec1[i].moved_from);
        container.template emplace<callable1>(vtable1, std::move(vec1[i]));
        ASSERT_FALSE(vec1[i].move_constructed);
        ASSERT_TRUE (vec1[i].moved_from);
      }

      std::vector<callable2> vec2;
      vec2.reserve(num2);
      for (size_t i = 0; i < num2; ++i) {
        vec2.emplace_back(make_type2(init_val2, i));
        ASSERT_FALSE(vec2[i].move_constructed);
        ASSERT_FALSE(vec2[i].moved_from);
        container.template emplace<callable2>(vtable2, std::move(vec2[i]));
        ASSERT_FALSE(vec2[i].move_constructed);
        ASSERT_TRUE (vec2[i].moved_from);
      }

      ASSERT_EQ(container.size(), num0+num1+num2);
      ASSERT_GE(container.storage_size(),
          num0*sizeof(callable0) +
          num1*sizeof(callable1) +
          num2*sizeof(callable2));
    };

    auto test_contents = [&](WorkStorage_type& container, double init_val0, double init_val1, double init_val2) {

      ASSERT_EQ(container.size(), num0+num1+num2);
      ASSERT_GE(container.storage_size(),
          num0*sizeof(callable0) +
          num1*sizeof(callable1) +
          num2*sizeof(callable2));

      {
        auto iter = container.begin();

        for (size_t i = 0; i < num0; ++i) {
          type0 val{};
          bool move_constructed = false;
          bool moved_from = true;
          WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

          type0 expected = make_type0(init_val0, i);
          ASSERT_EQ(val, expected);
          ASSERT_TRUE(move_constructed);
          ASSERT_FALSE(moved_from);

          ++iter;
        }

        for (size_t i = 0; i < num1; ++i) {
          type1 val{};
          bool move_constructed = false;
          bool moved_from = true;
          WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

          type1 expected = make_type1(init_val1, i);
          ASSERT_EQ(val, expected);
          ASSERT_TRUE(move_constructed);
          ASSERT_FALSE(moved_from);

          ++iter;
        }

        for (size_t i = 0; i < num2; ++i) {
          type2 val{};
          bool move_constructed = false;
          bool moved_from = true;
          WorkStruct_type::call(&*iter, (void*)&val, &move_constructed, &moved_from);

          type2 expected = make_type2(init_val2, i);
          ASSERT_EQ(val, expected);
          ASSERT_TRUE(move_constructed);
          ASSERT_FALSE(moved_from);

          ++iter;
        }

        ASSERT_EQ(iter, container.end());
      }
    };

    WorkStorage_type container(Allocator{});

    test_empty(container);
    fill_contents(container, 1.0, 100.0, 1000.0);

    container.clear();

    test_empty(container);
    fill_contents(container, 1.0, 100.0, 1000.0);
    test_contents(container, 1.0, 100.0, 1000.0);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_contents(container2, 1.0, 100.0, 1000.0);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_contents(container3, 1.0, 100.0, 1000.0);


    WorkStorage_type container4(Allocator{});

    fill_contents(container4, 1.5, 100.5, 1000.5);
    test_contents(container4, 1.5, 100.5, 1000.5);

    container4 = std::move(container3);

    test_empty(container3);
    test_contents(container4, 1.0, 100.0, 1000.0);

  }

  ASSERT_TRUE(success);
}


template <typename T>
class WorkGroupBasicWorkStorageConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageConstructorUnitTest);

template <typename T>
class WorkGroupBasicWorkStorageIteratorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageIteratorUnitTest);

template <typename T>
class WorkGroupBasicWorkStorageInsertCallUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageInsertCallUnitTest);

template <typename T>
class WorkGroupBasicWorkStorageMultipleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageMultipleUnitTest);


TYPED_TEST_P(WorkGroupBasicWorkStorageConstructorUnitTest, BasicWorkGroupWorkStorageConstructor)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageConstructor< StoragePolicy, Allocator >();
}

TYPED_TEST_P(WorkGroupBasicWorkStorageIteratorUnitTest, BasicWorkGroupWorkStorageIterator)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageIterator< StoragePolicy, Allocator >();
}

TYPED_TEST_P(WorkGroupBasicWorkStorageInsertCallUnitTest, BasicWorkGroupWorkStorageInsertCall)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageInsertCall< StoragePolicy, Allocator >();
}

TYPED_TEST_P(WorkGroupBasicWorkStorageMultipleUnitTest, BasicWorkGroupWorkStorageMultiple)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupWorkStorageMultiple< StoragePolicy, Allocator >(
      dist(rng), dist(rng), dist(rng));
}

#endif  //__TEST_WORKGROUP_WORKSTORAGE__
