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

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment, RAJA::RangeStrideSegment>;

constexpr const int TEST_VEC_LEN = 1024 * 1024 * 5;

using namespace RAJA;

static const double dinit_val = 0.1;
static const int iinit_val = 1;

class ReduceSumCUDA : public ::testing::Test
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

double* ReduceSumCUDA::dvalue = nullptr;
double* ReduceSumCUDA::rand_dvalue = nullptr;
int* ReduceSumCUDA::ivalue = nullptr;

const size_t block_size = 256;

CUDA_TEST_F(ReduceSumCUDA, staggered_sum)
{
  double* dvalue = ReduceSumCUDA::dvalue;

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

CUDA_TEST_F(ReduceSumCUDA, staggered_sum2)
{
  double* dvalue = ReduceSumCUDA::dvalue;

  double dtinit = 5.0;

  ReduceSum<cuda_reduce<block_size>, double> dsum0(5.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum1;
  ReduceSum<cuda_reduce<block_size>, double> dsum2(5.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum3;
  ReduceSum<cuda_reduce<block_size>, double> dsum4(5.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum5;
  ReduceSum<cuda_reduce<block_size>, double> dsum6(5.0);
  ReduceSum<cuda_reduce<block_size>, double> dsum7;
  
  dsum0.reset(0.0);
  dsum1.reset(dtinit * 1.0);
  dsum2.reset(0.0);
  dsum3.reset(dtinit * 3.0);
  dsum4.reset(0.0);
  dsum5.reset(dtinit * 5.0);
  dsum6.reset(0.0);
  dsum7.reset(dtinit * 7.0);
  
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

CUDA_TEST_F(ReduceSumCUDA, indexset_aligned)
{
  double* dvalue = ReduceSumCUDA::dvalue;
  int* ivalue = ReduceSumCUDA::ivalue;

  RangeSegment seg0(0, TEST_VEC_LEN / 2);
  RangeSegment seg1(TEST_VEC_LEN / 2, TEST_VEC_LEN);

  UnitIndexSet iset;
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

//
// test 3 runs 4 reductions (2 int, 2 double) over disjoint chunks
//        of the array using an indexset with four range segments
//        not aligned with warp boundaries to check that reduction
//        mechanics don't depend on any sort of special indexing.
//
CUDA_TEST_F(ReduceSumCUDA, indexset_noalign)
{
  double* dvalue = ReduceSumCUDA::dvalue;
  int* ivalue = ReduceSumCUDA::ivalue;


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

CUDA_TEST_F(ReduceSumCUDA, atomic_reduce)
{
  double* rand_dvalue = ReduceSumCUDA::rand_dvalue;

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

CUDA_TEST_F(ReduceSumCUDA, increasing_size)
{
  double* dvalue = ReduceSumCUDA::dvalue;

  double dtinit = 5.0;

  for (int size = block_size; size <= TEST_VEC_LEN; size+=block_size ) {

    ReduceSum<cuda_reduce<block_size, true>, double> dsum0(dtinit);

    forall<cuda_exec<block_size, true> >(0, size, [=] __device__(int i) {
      dsum0 += dvalue[i];
    });

    double base_chk_val = dinit_val * double(size);

    ASSERT_FLOAT_EQ(base_chk_val + dtinit, dsum0.get());
  }
}
