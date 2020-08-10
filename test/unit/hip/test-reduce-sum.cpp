//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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
#include "RAJA_gtest.hpp"

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment,
                                         RAJA::ListSegment,
                                         RAJA::RangeStrideSegment>;

constexpr const int TEST_VEC_LEN = 1024 * 1024 * 5;

using namespace RAJA;

static const double dinit_val = 0.1;
static const int iinit_val = 1;

class ReduceSumHIPUnitTest : public ::testing::Test
{
public:
  static void SetUpTestCase()
  {

    dvalue = (double *) malloc(sizeof(double) * TEST_VEC_LEN);
    hipMalloc((void **)&d_dvalue, sizeof(double) * TEST_VEC_LEN);

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      dvalue[i] = dinit_val;
    }
    hipMemcpy(d_dvalue, dvalue,
              sizeof(double) * TEST_VEC_LEN,
              hipMemcpyHostToDevice);

    ivalue = (int *) malloc(sizeof(int) * TEST_VEC_LEN);
    hipMalloc((void **)&d_ivalue, sizeof(int) * TEST_VEC_LEN);

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      ivalue[i] = iinit_val;
    }
    hipMemcpy(d_ivalue, ivalue,
              sizeof(int) * TEST_VEC_LEN,
              hipMemcpyHostToDevice);

    rand_dvalue = (double *) malloc(sizeof(double) * TEST_VEC_LEN);
    hipMalloc((void **)&d_rand_dvalue, sizeof(double) * TEST_VEC_LEN);
  }

  static void TearDownTestCase()
  {
    free(dvalue);
    free(rand_dvalue);
    free(ivalue);
    hipFree(d_dvalue);
    hipFree(d_rand_dvalue);
    hipFree(d_ivalue);
  }

  static double *dvalue;
  static double *d_dvalue;
  static double *rand_dvalue;
  static double *d_rand_dvalue;
  static int *ivalue;
  static int *d_ivalue;
};

double* ReduceSumHIPUnitTest::dvalue = nullptr;
double* ReduceSumHIPUnitTest::rand_dvalue = nullptr;
int* ReduceSumHIPUnitTest::ivalue = nullptr;
double* ReduceSumHIPUnitTest::d_dvalue = nullptr;
double* ReduceSumHIPUnitTest::d_rand_dvalue = nullptr;
int* ReduceSumHIPUnitTest::d_ivalue = nullptr;

const size_t block_size = 256;

GPU_TEST_F(ReduceSumHIPUnitTest, atomic_reduce)
{
  double* rand_dvalue = ReduceSumHIPUnitTest::rand_dvalue;
  double* d_rand_dvalue = ReduceSumHIPUnitTest::d_rand_dvalue;

  ReduceSum<hip_reduce_atomic, double> dsumN(0.0);
  ReduceSum<hip_reduce_atomic, double> dsumP(0.0);

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
    hipMemcpy(d_rand_dvalue, rand_dvalue,
              sizeof(double) * TEST_VEC_LEN,
              hipMemcpyHostToDevice);

    forall<hip_exec<block_size> >(RangeSegment(0, TEST_VEC_LEN),
                                   [=] RAJA_HOST_DEVICE(int i) {
                                     if (d_rand_dvalue[i] < 0.0) {
                                       dsumN += d_rand_dvalue[i];
                                     } else {
                                       dsumP += d_rand_dvalue[i];
                                     }
                                   });

    ASSERT_FLOAT_EQ(dsumN.get(), neg_chk_val);
    ASSERT_FLOAT_EQ(dsumP.get(), pos_chk_val);
  }
}

GPU_TEST_F(ReduceSumHIPUnitTest, increasing_size)
{
  double* dvalue = ReduceSumHIPUnitTest::d_dvalue;

  double dtinit = 5.0;

  for (int size = block_size; size <= TEST_VEC_LEN; size += block_size) {

    ReduceSum<hip_reduce, double> dsum0(dtinit);

    forall<hip_exec<block_size, true> >(RangeSegment(0, size),
                                         [=] RAJA_DEVICE(int i) {
                                           dsum0 += dvalue[i];
                                         });

    double base_chk_val = dinit_val * double(size);

    ASSERT_FLOAT_EQ(base_chk_val + dtinit, dsum0.get());
  }
}
