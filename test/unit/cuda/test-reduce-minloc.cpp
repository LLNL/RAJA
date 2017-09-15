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
/// Source file containing tests for RAJA GPU min-loc reductions.
///

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;

constexpr const RAJA::Index_type TEST_VEC_LEN = 1024 * 8;

static const int test_repeat = 10;
static const size_t block_size = 256;

// for setting random values in arrays
static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<double> dist(-10, 10);
static std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);

static void reset(double* ptr, long length, double def)
{
  for (long i = 0; i < length; ++i) {
    ptr[i] = def;
  }
}

template <typename T>
struct reduce_applier;
template <typename T, typename U>
struct reduce_applier<ReduceMinLoc<T, U>> {
  static U def() { return DBL_MAX; }
  static U big() { return -500.0; }
  RAJA_DEVICE static void apply(ReduceMinLoc<T, U> const& r,
                                U const& val,
                                Index_type i)
  {
    r.minloc(val, i);
  }
  template <bool B>
  RAJA_HOST_DEVICE static void apply(reduce::detail::ValueLoc<U, B>& l,
                                     reduce::detail::ValueLoc<U, B> const& r)
  {
    l = l > r ? r : l;
  }
  template <bool B>
  static void cmp(ReduceMinLoc<T, U>& l,
                  reduce::detail::ValueLoc<U, B> const& r)
  {
    ASSERT_FLOAT_EQ(r.val, l.get());
    ASSERT_EQ(r.loc, l.getLoc());
  }
};
template <typename T, typename U>
struct reduce_applier<ReduceMaxLoc<T, U>> {
  static U def() { return -DBL_MAX; }
  static U big() { return 500.0; }
  RAJA_DEVICE static void apply(ReduceMaxLoc<T, U> const& r,
                                U const& val,
                                Index_type i)
  {
    r.maxloc(val, i);
  }
  template <bool B>
  RAJA_HOST_DEVICE static void apply(reduce::detail::ValueLoc<U, B>& l,
                                     reduce::detail::ValueLoc<U, B> const& r)
  {
    l = l > r ? l : r;
  }
  template <bool B>
  static void cmp(ReduceMaxLoc<T, U>& l,
                                   reduce::detail::ValueLoc<U, B> const& r)
  {
    ASSERT_FLOAT_EQ(r.val, l.get());
    ASSERT_EQ(r.loc, l.getLoc());
  }
};

template <typename Reducer>
class ReduceCUDA : public ::testing::Test
{
  using applier = reduce_applier<Reducer>;

public:
  static double* dvalue;
  static void SetUpTestCase()
  {
    cudaMallocManaged((void**)&dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);
    reset(dvalue, TEST_VEC_LEN, applier::def());
  }
  static void TearDownTestCase() { cudaFree(dvalue); }
};

template <typename Reducer>
double* ReduceCUDA<Reducer>::dvalue = nullptr;


TYPED_TEST_CASE_P(ReduceCUDA);

CUDA_TYPED_TEST_P(ReduceCUDA, generic)
{

  using applier = reduce_applier<TypeParam>;
  using reducer = ReduceCUDA<TypeParam>;
  double* dvalue = reducer::dvalue;
  reset(dvalue, TEST_VEC_LEN, applier::def());

  reduce::detail::ValueLoc<double> dcurrentMin(applier::def(), -1);

  for (int tcount = 0; tcount < test_repeat; ++tcount) {


    TypeParam dmin0(applier::def(), -1);
    TypeParam dmin1(applier::def(), -1);
    TypeParam dmin2(applier::big(), -1);

    int loops = 16;
    for (int k = 0; k < loops; k++) {

      double droll = dist(mt);
      int index = int(dist2(mt));
      reduce::detail::ValueLoc<double> lmin{droll, index};
      dvalue[index] = droll;
      applier::apply(dcurrentMin, lmin);

      forall<cuda_exec<block_size>>(0, TEST_VEC_LEN, [=] __device__(int i) {
        applier::apply(dmin0, dvalue[i], i);
        applier::apply(dmin1, 2 * dvalue[i], i);
        applier::apply(dmin2, dvalue[i], i);
      });

      applier::cmp(dmin0, dcurrentMin);

      ASSERT_FLOAT_EQ(dcurrentMin.val * 2, dmin1.get());
      ASSERT_EQ(dcurrentMin.getLoc(), dmin1.getLoc());
      ASSERT_FLOAT_EQ(applier::big(), dmin2.get());
    }
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 2 runs 2 reductions over complete array using an indexset
//        with two range segments to check reduction object state
//        is maintained properly across kernel invocations.
//
CUDA_TYPED_TEST_P(ReduceCUDA, indexset_align)
{

  using applier = reduce_applier<TypeParam>;
  double* dvalue = ReduceCUDA<TypeParam>::dvalue;

  reset(dvalue, TEST_VEC_LEN, applier::def());

  reduce::detail::ValueLoc<double> dcurrentMin(applier::def(), -1);

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    RangeSegment seg0(0, TEST_VEC_LEN / 2);
    RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

    IndexSet iset;
    iset.push_back(seg0);
    iset.push_back(seg1);

    TypeParam dmin0(applier::def(), -1);
    TypeParam dmin1(applier::def(), -1);

    int index = int(dist2(mt));

    double droll = dist(mt);
    dvalue[index] = droll;
    reduce::detail::ValueLoc<double> lmin{droll, index};
    dvalue[index] = droll;
    applier::apply(dcurrentMin, lmin);

    forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
        iset, [=] __device__(int i) {
          applier::apply(dmin0, dvalue[i], i);
          applier::apply(dmin1, 2 * dvalue[i], i);
        });

    ASSERT_FLOAT_EQ(double(dcurrentMin), double(dmin0));
    ASSERT_FLOAT_EQ(2 * double(dcurrentMin), double(dmin1));
    ASSERT_EQ(dcurrentMin.getLoc(), dmin0.getLoc());
    ASSERT_EQ(dcurrentMin.getLoc(), dmin1.getLoc());
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 3 runs 2 reductions over disjoint chunks of the array using
//        an indexset with four range segments not aligned with
//        warp boundaries to check that reduction mechanics don't
//        depend on any sort of special indexing.
//
CUDA_TYPED_TEST_P(ReduceCUDA, indexset_noalign)
{

  using applier = reduce_applier<TypeParam>;
  double* dvalue = ReduceCUDA<TypeParam>::dvalue;

  RangeSegment seg0(1, 230);
  RangeSegment seg1(237, 385);
  RangeSegment seg2(860, 1110);
  RangeSegment seg3(2490, 4003);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    reset(dvalue, TEST_VEC_LEN, applier::def());

    reduce::detail::ValueLoc<double> dcurrentMin(applier::def(), -1);

    TypeParam dmin0(applier::def(), -1);
    TypeParam dmin1(applier::def(), -1);

    // pick an index in one of the segments
    int index = 97;                     // seg 0
    if (tcount % 2 == 0) index = 297;   // seg 1
    if (tcount % 3 == 0) index = 873;   // seg 2
    if (tcount % 4 == 0) index = 3457;  // seg 3

    double droll = dist(mt);
    dvalue[index] = droll;

    reduce::detail::ValueLoc<double> lmin{droll, index};
    dvalue[index] = droll;
    applier::apply(dcurrentMin, lmin);

    forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
        iset, [=] __device__(int i) {
          applier::apply(dmin0, dvalue[i], i);
          applier::apply(dmin1, 2 * dvalue[i], i);
        });

    ASSERT_FLOAT_EQ(dcurrentMin.val, double(dmin0));
    ASSERT_FLOAT_EQ(2 * dcurrentMin.val, double(dmin1));
    ASSERT_EQ(dcurrentMin.getLoc(), dmin0.getLoc());
    ASSERT_EQ(dcurrentMin.getLoc(), dmin1.getLoc());
  }
}

REGISTER_TYPED_TEST_CASE_P(ReduceCUDA,
                           generic,
                           indexset_align,
                           indexset_noalign);

using MinLocTypes =
    ::testing::Types<ReduceMinLoc<cuda_reduce<block_size>, double>>;
INSTANTIATE_TYPED_TEST_CASE_P(MinLoc, ReduceCUDA, MinLocTypes);

using MaxLocTypes =
    ::testing::Types<ReduceMaxLoc<cuda_reduce<block_size>, double>>;
INSTANTIATE_TYPED_TEST_CASE_P(MaxLoc, ReduceCUDA, MaxLocTypes);
