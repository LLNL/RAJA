//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic multipolicy operation
///

#include <cstddef>
#include "gtest/gtest.h"

// Tag type to dispatch to test bodies based on policy selected by multipolicy
template <int i>
struct mp_tag {
};

struct mp_test_body;
namespace RAJA
{
namespace impl
{
// fake RAJA::impl::forall overload to test multipolicy dispatch
template <int i, typename Iterable>
void forall(const mp_tag<i> &p, Iterable &&iter, mp_test_body const &body)
{
  body(p, iter.size());
}
}
}

// NOTE: this *must* be after the above to work
#include "RAJA/RAJA.hpp"

// This functor implements different test bodies depending on the mock "policy"
// selected by multipolicy, asserting the ranges of values that are selected in
// the MultiPolicy basic test below
struct mp_test_body {
  void operator()(mp_tag<1> const &, std::size_t size) const
  {
    ASSERT_LT(size, std::size_t{100});
  }
  void operator()(mp_tag<2> const &, std::size_t size) const
  {
    ASSERT_GT(size, std::size_t{99});
  }
  void operator()(mp_tag<3> const &, std::size_t size) const
  {
    ASSERT_GT(size, std::size_t{10});
    ASSERT_LT(size, std::size_t{99});
  }
};


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
