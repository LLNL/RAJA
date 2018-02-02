//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic multipolicy operation
///

#include <cstddef>
#include "gtest/gtest.h"

// Tag type to dispatch to test bodies based on policy selected by multipolicy

//struct mp_test_body;

namespace test_policy
{
template <int i>
struct mp_tag {
};
}

// This functor implements different test bodies depending on the mock "policy"
// selected by multipolicy, asserting the ranges of values that are selected in
// the MultiPolicy basic test below
struct mp_test_body {
  void operator()(test_policy::mp_tag<1> const &, std::size_t size) const
  {
    ASSERT_LT(size, std::size_t{100});
  }
  void operator()(test_policy::mp_tag<2> const &, std::size_t size) const
  {
    ASSERT_GT(size, std::size_t{99});
  }
  void operator()(test_policy::mp_tag<3> const &, std::size_t size) const
  {
    ASSERT_GT(size, std::size_t{10});
    ASSERT_LT(size, std::size_t{99});
  }
};

namespace test_policy
{
// fake forall_impl overload to test multipolicy dispatch
template <int i, typename Iterable>
void forall_impl(const mp_tag<i> &p, Iterable &&iter, mp_test_body const &body)
{
  body(p, iter.size());
}
}

using test_policy::mp_tag;

// NOTE: this *must* be after the above to work
#include "RAJA/RAJA.hpp"

TEST(MultiPolicy, basic)
{
  auto mp = RAJA::make_multi_policy<mp_tag<1>, mp_tag<2>>(
      [](const RAJA::RangeSegment &r) {
        if (r.size() < 100) {
          return 0;
        } else {
          return 1;
        }
      });
  RAJA::forall(mp, RAJA::RangeSegment(0, 5), mp_test_body{});
  RAJA::forall(mp, RAJA::RangeSegment(0, 101), mp_test_body{});
  // Nest a multipolicy to ensure value-based policies are preserved
  auto mp2 = RAJA::make_multi_policy(std::make_tuple(mp_tag<3>{}, mp),
                                     [](const RAJA::RangeSegment &r) {
                                       if (r.size() > 10 && r.size() < 90) {
                                         return 0;
                                       } else {
                                         return 1;
                                       }
                                     });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 5), mp_test_body{});
  RAJA::forall(mp2, RAJA::RangeSegment(0, 91), mp_test_body{});
  RAJA::forall(mp2, RAJA::RangeSegment(0, 50), mp_test_body{});
}

template <typename Multipolicy, typename Iterable>
void make_invalid_index_throw(Multipolicy &&mp, Iterable &&iter)
{
  RAJA::forall(mp, iter, [](RAJA::Index_type) {});
}

TEST(MultiPolicy, invalid_index)
{
  static constexpr const int limit = 100;
  RAJA::RangeSegment seg(0, limit);
  auto mp = RAJA::make_multi_policy<RAJA::seq_exec, RAJA::seq_exec>(
      [](const RAJA::RangeSegment &r) {
        if (r.size() < limit / 2) return 0;
        if (r.size() < limit) return 1;
        return 2;
      });
  ASSERT_THROW(make_invalid_index_throw(mp, seg), std::runtime_error);
}
