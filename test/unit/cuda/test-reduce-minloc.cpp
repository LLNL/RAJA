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

#include "gtest/gtest.h"
#include "RAJA/RAJA.hpp"

#define TEST_VEC_LEN 1024 * 1024 * 8

struct minloc_t {
  double val;
  int idx;
};

using namespace RAJA;
using namespace std;

const int test_repeat = 10;
const size_t block_size = 256;

// for setting random values in arrays
random_device rd;
mt19937 mt(rd());
uniform_real_distribution<double> dist(-10, 10);
uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);


double* initData()
{
  double* dvalue;

  cudaMallocManaged((void**)&dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    dvalue[i] = DBL_MAX;
  }
  return dvalue;
}

void test1()
{
  double* dvalue = initData();

  minloc_t dcurrentMin;
  dcurrentMin.val = DBL_MAX;
  dcurrentMin.idx = -1;

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    double BIG_MIN = -500.0;
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DBL_MAX, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DBL_MAX, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin2(BIG_MIN, -1);

    int loops = 16;
    for (int k = 0; k < loops; k++) {

      double droll = dist(mt);
      int index = int(dist2(mt));
      minloc_t lmin = {droll, index};
      dvalue[index] = droll;
      dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
        dmin0.minloc(dvalue[i], i);
        dmin1.minloc(2 * dvalue[i], i);
        dmin2.minloc(dvalue[i], i);
      });

      ASSERT_FLOAT_EQ(dcurrentMin.val, dmin0.get());
      ASSERT_FLOAT_EQ(dcurrentMin.val * 2, dmin1.get());
      ASSERT_FLOAT_EQ(BIG_MIN, dmin2.get());
      ASSERT_EQ(dcurrentMin.idx, dmin0.getLoc());
      ASSERT_EQ(dcurrentMin.idx, dmin1.getLoc());
    }
  }
  cudaFree(dvalue);
}

////////////////////////////////////////////////////////////////////////////

//
// test 2 runs 2 reductions over complete array using an indexset
//        with two range segments to check reduction object state
//        is maintained properly across kernel invocations.
//
void test2()
{
  double* dvalue = initData();

  minloc_t dcurrentMin;
  dcurrentMin.val = DBL_MAX;
  dcurrentMin.idx = -1;

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    RangeSegment seg0(0, TEST_VEC_LEN / 2);
    RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

    IndexSet iset;
    iset.push_back(seg0);
    iset.push_back(seg1);

    ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DBL_MAX, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DBL_MAX, -1);

    int index = int(dist2(mt));

    double droll = dist(mt);
    dvalue[index] = droll;
    minloc_t lmax = {droll, index};
    dvalue[index] = droll;
    dcurrentMin = RAJA_MINLOC(dcurrentMin, lmax);

    forall<ExecPolicy<seq_segit, cuda_exec<block_size> > >(
        iset, [=] __device__(int i) {
          dmin0.minloc(dvalue[i], i);
          dmin1.minloc(2 * dvalue[i], i);
        });

    ASSERT_FLOAT_EQ(dcurrentMin.val, double(dmin0));
    ASSERT_FLOAT_EQ(2 * dcurrentMin.val, double(dmin1));
    ASSERT_EQ(dcurrentMin.idx, dmin0.getLoc());
    ASSERT_EQ(dcurrentMin.idx, dmin1.getLoc());
  }
  cudaFree(dvalue);
}

////////////////////////////////////////////////////////////////////////////

//
// test 3 runs 2 reductions over disjoint chunks of the array using
//        an indexset with four range segments not aligned with
//        warp boundaries to check that reduction mechanics don't
//        depend on any sort of special indexing.
//
void test3()
{

  RangeSegment seg0(1, 1230);
  RangeSegment seg1(1237, 3385);
  RangeSegment seg2(4860, 10110);
  RangeSegment seg3(20490, 32003);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);


  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    double* dvalue = initData();

    minloc_t dcurrentMin;
    dcurrentMin.val = DBL_MAX;
    dcurrentMin.idx = -1;

    ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DBL_MAX, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DBL_MAX, -1);

    // pick an index in one of the segments
    int index = 897;                     // seg 0
    if (tcount % 2 == 0) index = 1297;   // seg 1
    if (tcount % 3 == 0) index = 7853;   // seg 2
    if (tcount % 4 == 0) index = 29457;  // seg 3

    double droll = dist(mt);
    dvalue[index] = droll;

    minloc_t lmax = {droll, index};
    dvalue[index] = droll;
    dcurrentMin = RAJA_MINLOC(dcurrentMin, lmax);

    forall<ExecPolicy<seq_segit, cuda_exec<block_size> > >(
        iset, [=] __device__(int i) {
          dmin0.minloc(dvalue[i], i);
          dmin1.minloc(2 * dvalue[i], i);
        });

    ASSERT_FLOAT_EQ(dcurrentMin.val, double(dmin0));
    ASSERT_FLOAT_EQ(2 * dcurrentMin.val, double(dmin1));
    ASSERT_EQ(dcurrentMin.idx, dmin0.getLoc());
    ASSERT_EQ(dcurrentMin.idx, dmin1.getLoc());

    cudaFree(dvalue);
  }
}

TEST(ReduceMinLoc, generic) { test1(); }

TEST(ReduceMinLoc, indexset_align) { test2(); }

TEST(ReduceMinLoc, indexset_noalign) { test3(); }
