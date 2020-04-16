//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

struct ForallViewCUDA : ::testing::Test {
  virtual void SetUp()
  {
    alen = 100000;
    test_val = 0.123;

    arr_h = (double*)allocate_aligned(DATA_ALIGN, alen * sizeof(double));

    for (Index_type i = 0; i < alen; ++i) {
      arr_h[i] = double(rand() % 65536);
    }

    cudaErrchk(cudaMalloc((void**)&arr_d, alen * sizeof(double)));

    cudaErrchk(cudaMemcpy(arr_d, arr_h, alen * sizeof(double), cudaMemcpyHostToDevice));
  }

  virtual void TearDown()
  {
    free_aligned(arr_h);
    cudaErrchk(cudaFree(arr_d));
  }
};

GPU_TEST_F(ForallViewCUDA, ForallViewOffsetLayout2D)
{

  using RAJA::Index_type;
  Index_type* box;
  const Index_type DIM = 2;
  const Index_type N = 2;
  const Index_type boxSize = (N + 2) * (N + 2);

  cudaErrchk(cudaMallocManaged((void**)&box,
                    boxSize * sizeof(Index_type),
                    cudaMemAttachGlobal));

  RAJA::OffsetLayout<DIM> layout =
      RAJA::make_offset_layout<DIM>({{-1, -1}}, {{2, 2}});
  RAJA::View<Index_type, RAJA::OffsetLayout<DIM>> boxview(box, layout);

  forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, N * N),
                               [=] RAJA_HOST_DEVICE(Index_type i) {
                                 const int col = i % N;
                                 const int row = i / N;
                                 boxview(row, col) = 1000;
                               });


  for (Index_type row = 0; row < N; ++row) {
    for (Index_type col = 0; col < N; ++col) {
      int id = (col + 1) + (N + 2) * (row + 1);
      EXPECT_EQ(box[id], 1000);
    }
  }

  cudaErrchk(cudaFree(box));
}
