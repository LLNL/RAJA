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

#include <cstdlib>

#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;
using namespace std;

const size_t block_size = 256;

static double* arr_h;
static double* arr_d;
static Index_type alen;
static double test_val;

struct ForallViewCUDA : ::testing::Test
{
  virtual void SetUp()
  {
    alen = 100000;
    test_val = 0.123;

    arr_h = (double*) allocate_aligned(DATA_ALIGN, alen * sizeof(double));

    for (Index_type i = 0; i < alen; ++i) {
      arr_h[i] = double(rand() % 65536);
    }

    cudaErrchk( cudaMalloc( (void**)&arr_d, alen * sizeof(double) ) );

    cudaErrchk( cudaMemcpy( arr_d, arr_h,
                            alen * sizeof(double),
                            cudaMemcpyHostToDevice ) );
  }

  virtual void TearDown()
  {
    free_aligned(arr_h);
    cudaErrchk( cudaFree(arr_d) );
  }
};

CUDA_TEST_F(ForallViewCUDA, ForallViewLayout)
{
  const Index_type alen = ::alen;
  double* arr_h = ::arr_h;
  double* arr_d = ::arr_d;
  double test_val = ::test_val;

  const RAJA::Layout<1> my_layout(alen);
  RAJA::View<double, RAJA::Layout<1> > view(arr_d, my_layout);

  forall<RAJA::cuda_exec<block_size>>(RAJA::RangeSegment(0, alen), 
    [=] __device__ (Index_type i) {
    view(i) = test_val;
  });

  cudaErrchk( cudaMemcpy( arr_h, arr_d,
              alen * sizeof(double),
              cudaMemcpyDeviceToHost ) );

  for (Index_type i = 0; i < alen; ++i) {
    EXPECT_EQ(arr_h[i], test_val);
  }
}

CUDA_TEST_F(ForallViewCUDA, ForallViewOffsetLayout)
{
  const Index_type alen = ::alen;
  double* arr_h = ::arr_h;
  double* arr_d = ::arr_d;
  double test_val = ::test_val;

  RAJA::OffsetLayout<1> my_layout = 
                        RAJA::make_offset_layout<1>({{1}}, {{alen+1}}); 
  RAJA::View<double, RAJA::OffsetLayout<1> > view(arr_d, my_layout);

  forall<RAJA::cuda_exec<block_size>>(RAJA::RangeSegment(1, alen+1), 
  [=] __device__(Index_type i) { 
    view(i) = test_val;
  });

  cudaErrchk( cudaMemcpy( arr_h, arr_d,
              alen * sizeof(double),
              cudaMemcpyDeviceToHost ) );

  for (Index_type i = 0; i < alen; ++i) {
    EXPECT_EQ(arr_h[i], test_val);
  }
}
