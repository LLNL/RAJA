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
                                         RAJA::RangeStrideSegment>;

constexpr const int TEST_VEC_LEN = 1024 * 1024 * 5;

using namespace RAJA;

static const double dinit_val = 0.1;
static const int iinit_val = 1;

class ReduceSumCUDA : public ::testing::Test
{
public:
  static void SetUpTestCase()
  {

    cudaErrchk(cudaMallocManaged((void**)&dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal));

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      dvalue[i] = dinit_val;
    }

    cudaErrchk(cudaMallocManaged((void**)&ivalue,
                      sizeof(int) * TEST_VEC_LEN,
                      cudaMemAttachGlobal));

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      ivalue[i] = iinit_val;
    }

    cudaErrchk(cudaMallocManaged((void**)&rand_dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal));
  }

  static void TearDownTestCase()
  {
    cudaErrchk(cudaFree(dvalue));
    cudaErrchk(cudaFree(rand_dvalue));
    cudaErrchk(cudaFree(ivalue));
  }

  static double* dvalue;
  static double* rand_dvalue;
  static int* ivalue;
};

double* ReduceSumCUDA::dvalue = nullptr;
double* ReduceSumCUDA::rand_dvalue = nullptr;
int* ReduceSumCUDA::ivalue = nullptr;

const size_t block_size = 256;

GPU_TEST_F(ReduceSumCUDA, atomic_reduce)
{
  double* rand_dvalue = ReduceSumCUDA::rand_dvalue;

  ReduceSum<cuda_reduce_atomic, double> dsumN(0.0);
  ReduceSum<cuda_reduce_atomic, double> dsumP(0.0);

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
    forall<cuda_exec<block_size> >(RangeSegment(0, TEST_VEC_LEN),
                                   [=] RAJA_HOST_DEVICE(int i) {
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

GPU_TEST_F(ReduceSumCUDA, increasing_size)
{
  double* dvalue = ReduceSumCUDA::dvalue;

  double dtinit = 5.0;

  for (int size = block_size; size <= TEST_VEC_LEN; size += block_size) {

    ReduceSum<cuda_reduce, double> dsum0(dtinit);

    forall<cuda_exec<block_size, true> >(RangeSegment(0, size),
                                         [=] RAJA_DEVICE(int i) {
                                           dsum0 += dvalue[i];
                                         });

    double base_chk_val = dinit_val * double(size);

    ASSERT_FLOAT_EQ(base_chk_val + dtinit, dsum0.get());
  }
}
