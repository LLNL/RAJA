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

struct ForallViewHIP : ::testing::Test {
  virtual void SetUp()
  {
    alen = 100000;
    test_val = 0.123;

    arr_h = (double*)allocate_aligned(DATA_ALIGN, alen * sizeof(double));

    for (Index_type i = 0; i < alen; ++i) {
      arr_h[i] = double(rand() % 65536);
    }

    hipErrchk(hipMalloc((void**)&arr_d, alen * sizeof(double)));

    hipErrchk(hipMemcpy(
        arr_d, arr_h, alen * sizeof(double), hipMemcpyHostToDevice));
  }

  virtual void TearDown()
  {
    free_aligned(arr_h);
    hipErrchk(hipFree(arr_d));
  }
};

GPU_TEST_F(ForallViewHIP, ForallViewLayout)
{
  const Index_type alen = ::alen;
  double* arr_h = ::arr_h;
  double* arr_d = ::arr_d;
  double test_val = ::test_val;

  const RAJA::Layout<1> my_layout(alen);
  RAJA::View<double, RAJA::Layout<1>> view(arr_d, my_layout);

  forall<RAJA::hip_exec<block_size>>(RAJA::RangeSegment(0, alen),
                                      [=] RAJA_HOST_DEVICE(Index_type i) {
                                        view(i) = test_val;
                                      });

  hipErrchk(
      hipMemcpy(arr_h, arr_d, alen * sizeof(double), hipMemcpyDeviceToHost));

  for (Index_type i = 0; i < alen; ++i) {
    EXPECT_EQ(arr_h[i], test_val);
  }
}

GPU_TEST_F(ForallViewHIP, ForallViewOffsetLayout)
{
  const Index_type alen = ::alen;
  double* arr_h = ::arr_h;
  double* arr_d = ::arr_d;
  double test_val = ::test_val;

  RAJA::OffsetLayout<1> my_layout =
      RAJA::make_offset_layout<1>({{1}}, {{alen + 1}});
  RAJA::View<double, RAJA::OffsetLayout<1>> view(arr_d, my_layout);

  forall<RAJA::hip_exec<block_size>>(RAJA::RangeSegment(1, alen + 1),
                                      [=] RAJA_DEVICE(Index_type i) {
                                        view(i) = test_val;
                                      });

  hipErrchk(
      hipMemcpy(arr_h, arr_d, alen * sizeof(double), hipMemcpyDeviceToHost));

  for (Index_type i = 0; i < alen; ++i) {
    EXPECT_EQ(arr_h[i], test_val);
  }
}

GPU_TEST_F(ForallViewHIP, ForallViewOffsetLayout2D)
{

  using RAJA::Index_type;
  Index_type* box;
  Index_type *d_box;
  const Index_type DIM = 2;
  const Index_type N = 2;
  const Index_type boxSize = (N + 2) * (N + 2);

  box = (Index_type *) malloc(boxSize*sizeof(Index_type));
  hipMalloc((void**)&d_box, boxSize*sizeof(Index_type));

  RAJA::OffsetLayout<DIM> layout =
      RAJA::make_offset_layout<DIM>({{-1, -1}}, {{2, 2}});
  RAJA::View<Index_type, RAJA::OffsetLayout<DIM>> boxview(d_box, layout);

  forall<RAJA::hip_exec<256>>(RAJA::RangeSegment(0, N * N),
                               [=] RAJA_HOST_DEVICE(Index_type i) {
                                 const int col = i % N;
                                 const int row = i / N;
                                 boxview(row, col) = 1000;
                               });

  hipErrchk( hipMemcpy( box, d_box,
              boxSize*sizeof(Index_type),
              hipMemcpyDeviceToHost ) );

  for (Index_type row = 0; row < N; ++row) {
    for (Index_type col = 0; col < N; ++col) {
      int id = (col + 1) + (N + 2) * (row + 1);
      EXPECT_EQ(box[id], 1000);
    }
  }

  hipFree(box);
}
