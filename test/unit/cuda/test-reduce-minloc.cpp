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

constexpr const RAJA::Index_type TEST_VEC_LEN = 1024 * 1024 * 8;

const int test_repeat = 10;
const size_t block_size = 256;
const double DEFAULT_VAL = DBL_MAX;
const double BIG_VAL = -500.0;

// for setting random values in arrays
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(-10, 10);
std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);

struct minloc_t {
  double val;
  int idx;
};

void reset(double* ptr, long length)
{
  for (long i = 0; i < length; ++i) {
    ptr[i] = DEFAULT_VAL;
  }
}

class ReduceMinLocTest : public ::testing::Test {
public:
  static double* dvalue;
  static void SetUpTestCase()
  {
    cudaMallocManaged((void**)&dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);
    reset(dvalue, TEST_VEC_LEN);
  }
  static void TearDownTestCase() { cudaFree(dvalue); }
};

double* ReduceMinLocTest::dvalue = nullptr;

CUDA_TEST_F(ReduceMinLocTest, generic)
{

  double* dvalue = ReduceMinLocTest::dvalue;
  reset(dvalue, TEST_VEC_LEN);

  minloc_t dcurrentMin;
  dcurrentMin.val = DEFAULT_VAL;
  dcurrentMin.idx = -1;

  for (int tcount = 0; tcount < test_repeat; ++tcount) {


    ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DEFAULT_VAL, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DEFAULT_VAL, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin2(BIG_VAL, -1);

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
      ASSERT_FLOAT_EQ(BIG_VAL, dmin2.get());
      ASSERT_EQ(dcurrentMin.idx, dmin0.getLoc());
      ASSERT_EQ(dcurrentMin.idx, dmin1.getLoc());
    }
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 2 runs 2 reductions over complete array using an indexset
//        with two range segments to check reduction object state
//        is maintained properly across kernel invocations.
//
CUDA_TEST_F(ReduceMinLocTest, indexset_align)
{

  double* dvalue = ReduceMinLocTest::dvalue;

  reset(dvalue, TEST_VEC_LEN);

  minloc_t dcurrentMin;
  dcurrentMin.val = DEFAULT_VAL;
  dcurrentMin.idx = -1;

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    RangeSegment seg0(0, TEST_VEC_LEN / 2);
    RangeSegment seg1(TEST_VEC_LEN / 2 + 1, TEST_VEC_LEN);

    IndexSet iset;
    iset.push_back(seg0);
    iset.push_back(seg1);

    ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DEFAULT_VAL, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DEFAULT_VAL, -1);

    int index = int(dist2(mt));

    double droll = dist(mt);
    dvalue[index] = droll;
    minloc_t lmin = {droll, index};
    dvalue[index] = droll;
    dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

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
}

////////////////////////////////////////////////////////////////////////////

//
// test 3 runs 2 reductions over disjoint chunks of the array using
//        an indexset with four range segments not aligned with
//        warp boundaries to check that reduction mechanics don't
//        depend on any sort of special indexing.
//
CUDA_TEST_F(ReduceMinLocTest, indexset_noalign)
{

  double* dvalue = ReduceMinLocTest::dvalue;

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

    reset(dvalue, TEST_VEC_LEN);

    minloc_t dcurrentMin;
    dcurrentMin.val = DEFAULT_VAL;
    dcurrentMin.idx = -1;

    ReduceMinLoc<cuda_reduce<block_size>, double> dmin0(DEFAULT_VAL, -1);
    ReduceMinLoc<cuda_reduce<block_size>, double> dmin1(DEFAULT_VAL, -1);

    // pick an index in one of the segments
    int index = 897;                     // seg 0
    if (tcount % 2 == 0) index = 1297;   // seg 1
    if (tcount % 3 == 0) index = 7853;   // seg 2
    if (tcount % 4 == 0) index = 29457;  // seg 3

    double droll = dist(mt);
    dvalue[index] = droll;

    minloc_t lmin = {droll, index};
    dvalue[index] = droll;
    dcurrentMin = RAJA_MINLOC(dcurrentMin, lmin);

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
}
