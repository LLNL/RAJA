//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU min reductions.
///

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment,
                                         RAJA::ListSegment,
                                         RAJA::RangeStrideSegment>;

constexpr const RAJA::Index_type TEST_VEC_LEN = 1024 * 1024 * 8;

static const int test_repeat = 10;
static const size_t block_size = 256;
static const double DEFAULT_VAL = DBL_MAX;
static const double BIG_VAL = -500.0;

// for setting random values in arrays
static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<double> dist(-10, 10);
static std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);

static void reset(double* ptr, long length)
{
  for (long i = 0; i < length; ++i) {
    ptr[i] = DEFAULT_VAL;
  }
}

class ReduceMinHIP : public ::testing::Test
{
public:
  static double* dvalue;
  static double* d_dvalue;

  static void SetUpTestCase()
  {
    dvalue = (double*) malloc(sizeof(double) * TEST_VEC_LEN);
    hipMalloc((void**)&d_dvalue,
                      sizeof(double) * TEST_VEC_LEN);
    reset(dvalue, TEST_VEC_LEN);
  }
  static void TearDownTestCase() { free(dvalue); hipFree(d_dvalue); }
};

double* ReduceMinHIP::dvalue = nullptr;
double* ReduceMinHIP::d_dvalue = nullptr;

GPU_TEST_F(ReduceMinHIP, generic)
{

  double* dvalue = ReduceMinHIP::dvalue;
  double* d_dvalue = ReduceMinHIP::d_dvalue;
  reset(dvalue, TEST_VEC_LEN);
  hipMemcpy(d_dvalue, dvalue, sizeof(double) * TEST_VEC_LEN, hipMemcpyHostToDevice);

  double dcurrentMin = DEFAULT_VAL;

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    ReduceMin<hip_reduce, double> dmin0; dmin0.reset(DEFAULT_VAL);
    ReduceMin<hip_reduce, double> dmin1(DEFAULT_VAL);
    ReduceMin<hip_reduce, double> dmin2(BIG_VAL);

    int loops = 16;
    for (int k = 0; k < loops; k++) {

      double droll = dist(mt);
      int index = int(dist2(mt));
      if (dvalue[index] > droll) {
        dvalue[index] = droll;
        dcurrentMin = RAJA_MIN(dcurrentMin, droll);
      }
      hipMemcpy(d_dvalue, dvalue, sizeof(double) * TEST_VEC_LEN, hipMemcpyHostToDevice);

      forall<hip_exec<block_size> >(RangeSegment(0, TEST_VEC_LEN),
                                     [=] RAJA_DEVICE(int i) {
                                       dmin0.min(d_dvalue[i]);
                                       dmin1.min(2 * d_dvalue[i]);
                                       dmin2.min(d_dvalue[i]);
                                     });

      ASSERT_FLOAT_EQ(dcurrentMin, dmin0.get());
      ASSERT_FLOAT_EQ(dcurrentMin * 2, dmin1.get());
      ASSERT_FLOAT_EQ(BIG_VAL, dmin2.get());
    }

    dmin0.reset(DEFAULT_VAL);
    dmin1.reset(DEFAULT_VAL);
    dmin2.reset(BIG_VAL);

    loops = 16;
    for (int k = 0; k < loops; k++) {

      double droll = dist(mt);
      int index = int(dist2(mt));
      if (dvalue[index] > droll) {
        dvalue[index] = droll;
        dcurrentMin = RAJA_MIN(dcurrentMin, droll);
      }
      hipMemcpy(d_dvalue, dvalue, sizeof(double) * TEST_VEC_LEN, hipMemcpyHostToDevice);

      forall<hip_exec<block_size> >(RangeSegment(0, TEST_VEC_LEN),
                                     [=] RAJA_HOST_DEVICE(int i) {
                                       dmin0.min(d_dvalue[i]);
                                       dmin1.min(2 * d_dvalue[i]);
                                       dmin2.min(d_dvalue[i]);
                                     });

      ASSERT_FLOAT_EQ(dcurrentMin, dmin0.get());
      ASSERT_FLOAT_EQ(dcurrentMin * 2, dmin1.get());
      ASSERT_FLOAT_EQ(BIG_VAL, dmin2.get());
    }
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 2 runs 2 reductions over complete array using an indexset
//        with two range segments to check reduction object state
//        is maintained properly across kernel invocations.
//
GPU_TEST_F(ReduceMinHIP, indexset_align)
{

  double* dvalue = ReduceMinHIP::dvalue;
  double* d_dvalue = ReduceMinHIP::d_dvalue;

  reset(dvalue, TEST_VEC_LEN);
  hipMemcpy(d_dvalue, dvalue, sizeof(double) * TEST_VEC_LEN, hipMemcpyHostToDevice);

  double dcurrentMin = DEFAULT_VAL;

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    RangeSegment seg0(0, TEST_VEC_LEN / 2);
    RangeSegment seg1(TEST_VEC_LEN / 2, TEST_VEC_LEN);

    UnitIndexSet iset;
    iset.push_back(seg0);
    iset.push_back(seg1);

    ReduceMin<hip_reduce, double> dmin0(DEFAULT_VAL);
    ReduceMin<hip_reduce, double> dmin1(DEFAULT_VAL);

    double droll = dist(mt);
    int index = int(dist2(mt));
    if (dvalue[index] > droll) {
      dvalue[index] = droll;
      dcurrentMin = RAJA_MIN(dcurrentMin, droll);
    }
    hipMemcpy(d_dvalue, dvalue, sizeof(double) * TEST_VEC_LEN, hipMemcpyHostToDevice);

    forall<ExecPolicy<seq_segit, hip_exec<block_size> > >(
        iset, [=] RAJA_DEVICE(int i) {
          dmin0.min(d_dvalue[i]);
          dmin1.min(2 * d_dvalue[i]);
        });

    ASSERT_FLOAT_EQ(dcurrentMin, double(dmin0));
    ASSERT_FLOAT_EQ(2 * dcurrentMin, double(dmin1));
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 3 runs 2 reductions over disjoint chunks of the array using
//        an indexset with four range segments not aligned with
//        warp boundaries to check that reduction mechanics don't
//        depend on any sort of special indexing.
//
GPU_TEST_F(ReduceMinHIP, indexset_noalign)
{

  double* dvalue = ReduceMinHIP::dvalue;
  double* d_dvalue = ReduceMinHIP::d_dvalue;

  RangeSegment seg0(1, 1230);
  RangeSegment seg1(1237, 3385);
  RangeSegment seg2(4860, 10110);
  RangeSegment seg3(20490, 32003);

  UnitIndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    reset(dvalue, TEST_VEC_LEN);

    double dcurrentMin = DEFAULT_VAL;

    ReduceMin<hip_reduce, double> dmin0(DEFAULT_VAL);
    ReduceMin<hip_reduce, double> dmin1(DEFAULT_VAL);

    // pick an index in one of the segments
    int index = 897;                     // seg 0
    if (tcount % 2 == 0) index = 1297;   // seg 1
    if (tcount % 3 == 0) index = 7853;   // seg 2
    if (tcount % 4 == 0) index = 29457;  // seg 3

    double droll = dist(mt);
    if (dvalue[index] > droll) {
      dvalue[index] = droll;
      dcurrentMin = RAJA_MIN(dcurrentMin, droll);
    }
    hipMemcpy(d_dvalue, dvalue, sizeof(double) * TEST_VEC_LEN, hipMemcpyHostToDevice);


    forall<ExecPolicy<seq_segit, hip_exec<block_size> > >(
        iset, [=] RAJA_HOST_DEVICE(int i) {
          dmin0.min(d_dvalue[i]);
          dmin1.min(2 * d_dvalue[i]);
        });

    ASSERT_FLOAT_EQ(dcurrentMin, double(dmin0));
    ASSERT_FLOAT_EQ(2 * dcurrentMin, double(dmin1));
  }
}
