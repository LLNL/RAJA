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
/// Source file containing tests for RAJA GPU reductions.
///

#include <math.h>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

constexpr const int TEST_VEC_LEN = 1024 * 1024 * 5;

using namespace RAJA;

const double dinit_val = 0.1;
const int iinit_val = 1;

class ReduceSumTest : public ::testing::Test
{
public:
  static void SetUpTestCase()
  {

    cudaMallocManaged((void **)&dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      dvalue[i] = dinit_val;
    }

    cudaMallocManaged((void **)&ivalue,
                      sizeof(int) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      ivalue[i] = iinit_val;
    }

    cudaMallocManaged((void **)&rand_dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal);
  }

  static void TearDownTestCase()
  {

    cudaFree(dvalue);
    cudaFree(rand_dvalue);
    cudaFree(ivalue);
  }

  static double *dvalue;
  static double *rand_dvalue;
  static int *ivalue;
};

double* ReduceSumTest::dvalue = nullptr;
double* ReduceSumTest::rand_dvalue = nullptr;
int* ReduceSumTest::ivalue = nullptr;

const size_t block_size = 256;

void test1(double* dvalue) {

  double dtinit = 5.0;

  ReduceSum<cuda_reduce<block_size>, double> dsum0(0.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum1(dtinit * 1.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum2(0.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum3(dtinit * 3.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum4(0.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum5(dtinit * 5.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum6(0.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum7(dtinit * 7.0);

  int loops = 2;
  for (int k = 0; k < loops; k++) {

    forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
      dsum0 += dvalue[i];
      dsum1 += dvalue[i] * 2.0;
      dsum2 += dvalue[i] * 3.0;
      dsum3 += dvalue[i] * 4.0;
      dsum4 += dvalue[i] * 5.0;
      dsum5 += dvalue[i] * 6.0;
      dsum6 += dvalue[i] * 7.0;
      dsum7 += dvalue[i] * 8.0;
    });

    double base_chk_val = dinit_val * double(TEST_VEC_LEN) * (k + 1);

    ASSERT_FLOAT_EQ(1 * base_chk_val, dsum0.get());
    ASSERT_FLOAT_EQ(2 * base_chk_val + (dtinit * 1.0), dsum1.get());
    ASSERT_FLOAT_EQ(3 * base_chk_val, dsum2.get());
    ASSERT_FLOAT_EQ(4 * base_chk_val + (dtinit * 3.0), dsum3.get());
    ASSERT_FLOAT_EQ(5 * base_chk_val, dsum4.get());
    ASSERT_FLOAT_EQ(6 * base_chk_val + (dtinit * 5.0), dsum5.get());
    ASSERT_FLOAT_EQ(7 * base_chk_val, dsum6.get());
    ASSERT_FLOAT_EQ(8 * base_chk_val + (dtinit * 7.0), dsum7.get());
  }
}

void test2(double* dvalue, int* ivalue) {

  RangeSegment seg0(0, TEST_VEC_LEN / 2);
  RangeSegment seg1(TEST_VEC_LEN / 2, TEST_VEC_LEN);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);

  double dtinit = 5.0;
  int itinit = 4;

  ReduceSum<cuda_reduce<block_size>, double> dsum0(dtinit * 1.0);
  ReduceSum<cuda_reduce<block_size>, int> isum1(itinit * 2);
  ReduceSum<cuda_reduce<block_size>, double> dsum2(dtinit * 3.0);
  ReduceSum<cuda_reduce<block_size>, int> isum3(itinit * 4);

  forallN<NestedPolicy<ExecList<ExecPolicy<seq_segit,
                                           cuda_exec<block_size> > > > >(
      iset, [=] __device__(int i) {
        dsum0 += dvalue[i];
        isum1 += 2 * ivalue[i];
        dsum2 += 3 * dvalue[i];
        isum3 += 4 * ivalue[i];
      });

  double dbase_chk_val = dinit_val * double(iset.getLength());
  int ibase_chk_val = iinit_val * (iset.getLength());

  ASSERT_FLOAT_EQ(dbase_chk_val + (dtinit * 1.0), dsum0.get());
  ASSERT_EQ(2 * ibase_chk_val + (itinit * 2), isum1.get());
  ASSERT_FLOAT_EQ(3 * dbase_chk_val + (dtinit * 3.0), dsum2.get());
  ASSERT_EQ(4 * ibase_chk_val + (itinit * 4), isum3.get());
}


////////////////////////////////////////////////////////////////////////////

//
// test 3 runs 4 reductions (2 int, 2 double) over disjoint chunks
//        of the array using an indexset with four range segments
//        not aligned with warp boundaries to check that reduction
//        mechanics don't depend on any sort of special indexing.
//
void test3(double* dvalue, int* ivalue) {

  RangeSegment seg0(1, 1230);
  RangeSegment seg1(1237, 3385);
  RangeSegment seg2(4860, 10110);
  RangeSegment seg3(20490, 32003);

  IndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  double dtinit = 5.0;
  int itinit = 4;

  ReduceSum<cuda_reduce<block_size>, double> dsum0(dtinit * 1.0);
  ReduceSum<cuda_reduce<block_size>, int> isum1(itinit * 2);
  ReduceSum<cuda_reduce<block_size>, double> dsum2(dtinit * 3.0);
  ReduceSum<cuda_reduce<block_size>, int> isum3(itinit * 4);

  forall<ExecPolicy<seq_segit, cuda_exec<block_size> > >(
      iset, [=] __device__(int i) {
        dsum0 += dvalue[i];
        isum1 += 2 * ivalue[i];
        dsum2 += 3 * dvalue[i];
        isum3 += 4 * ivalue[i];
      });

  double dbase_chk_val = dinit_val * double(iset.getLength());
  int ibase_chk_val = iinit_val * double(iset.getLength());

  ASSERT_FLOAT_EQ(double(dsum0), dbase_chk_val + (dtinit * 1.0));
  ASSERT_EQ(int(isum1), 2 * ibase_chk_val + (itinit * 2));
  ASSERT_FLOAT_EQ(double(dsum2), 3 * dbase_chk_val + (dtinit * 3.0));
  ASSERT_EQ(int(isum3), 4 * ibase_chk_val + (itinit * 4));

}

void test4(double* rand_dvalue) {

  ReduceSum<cuda_reduce_atomic<block_size>, double> dsumN(0.0);
  ReduceSum<cuda_reduce_atomic<block_size>, double> dsumP(0.0);

  double neg_chk_val = 0.0;
  double pos_chk_val = 0.0;

  int loops = 3;

  for (int k = 0; k < loops; k++) {

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      rand_dvalue[i] = drand48() - 0.5;
      if (rand_dvalue[i] < 0.0) {
        neg_chk_val += rand_dvalue[i];
      } else {
        pos_chk_val += rand_dvalue[i];
      }
    }
    forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
      if (rand_dvalue[i] < 0.0) {
        dsumN += rand_dvalue[i];
      } else {
        dsumP += rand_dvalue[i];
      }
    });

    ASSERT_FLOAT_EQ(dsumN.get(), neg_chk_val);
    ASSERT_FLOAT_EQ(dsumP.get(), pos_chk_val);
  }

}

TEST_F(ReduceSumTest, staggered_sum)
{
  test1(ReduceSumTest::dvalue);
}

TEST_F(ReduceSumTest, indexset_aligned)
{
  test2(ReduceSumTest::dvalue, ReduceSumTest::ivalue);
}

TEST_F(ReduceSumTest, indexset_noalign)
{
  test3(ReduceSumTest::dvalue, ReduceSumTest::ivalue);
}

TEST_F(ReduceSumTest, atomic_reduce)
{
  test4(ReduceSumTest::rand_dvalue);
}
