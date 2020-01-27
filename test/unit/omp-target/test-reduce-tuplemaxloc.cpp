//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA omp-target tuple reduction maxloc operations.
/// Separated from minloc to reduce XL compilation time.
///

#include "gtest/gtest.h"

#include <iostream>
#include "RAJA/RAJA.hpp"

#include <tuple>

struct Index2D {
  RAJA::Index_type idx, idy;
   constexpr Index2D() : idx(-1), idy(-1) {}
   constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
};

template <typename TUPLE>
class ReductionTupleLocTestTargetOMP : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    // 2 dimensional, 10x10
    array_length = 102;
    xdim = 10;
    ydim = 10;

    array = RAJA::allocate_aligned_type<double *>(RAJA::DATA_ALIGN,
                                                ydim * sizeof(double *));
    data = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                               array_length * sizeof(double));

    // set rows to point to data
    for ( int ii = 0; ii < ydim; ++ii ) {
      array[ii] = data + ii * ydim;
    }

    // setting data values
    int count = 0;
    for ( int ii = 0; ii < ydim; ++ii ) {
      for ( int jj = 0; jj < xdim; ++jj ) {
        array[ii][jj] = (RAJA::Real_type)(count++);
      }
    }

    array[ydim-1][xdim-1] = -1.0;
#pragma omp target enter data map(to : data[:array_length])
#pragma omp target enter data map(to : array[:ydim])

    sum = 0.0;
    min = array_length * 2;
    max = 0.0;
    minlocx = -1;
    minlocy = -1;
    maxlocx = -1;
    maxlocy = -1;

    for (int y = 0; y < ydim; ++y) {
      for ( int x = 0; x < xdim; ++x ) {
        RAJA::Real_type val = array[y][x];

        sum += val;

        if (val > max) {
          max = val;
          maxlocx = x;
          maxlocy = y;
        }

        if (val < min) {
          min = val;
          minlocx = x;
          minlocy = y;
        }
      }
    }
  }

  virtual void TearDown()
  {
    // NOTE: clang prefers cast to void * and fails compilation otherwise.
    // gcc prefers RAJA::Real_ptr * (with no compilation warnings).
#pragma omp target exit data map(release : array[:ydim])
    RAJA::free_aligned((void *)array);

#pragma omp target exit data map(release : data[:array_length])
    RAJA::free_aligned(data);
  }

  RAJA::Real_ptr * array;
  RAJA::Real_ptr data;

  RAJA::Real_type max;
  RAJA::Real_type min;
  RAJA::Real_type sum;
  RAJA::Real_type maxlocx;
  RAJA::Real_type maxlocy;
  RAJA::Real_type minlocx;
  RAJA::Real_type minlocy;

  RAJA::Index_type array_length;
  RAJA::Index_type xdim;
  RAJA::Index_type ydim;
};
TYPED_TEST_SUITE_P(ReductionTupleLocTestTargetOMP);

TYPED_TEST_P(ReductionTupleLocTestTargetOMP, ReduceMaxLocIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  RAJA::ReduceMaxLoc<RAJA::omp_target_reduce, double, Index2D> maxloc_reducer(-1024.0, Index2D(0, 0));

  auto actualdata = this->data;
  auto indirect = this->array;
  RAJA::forall<ExecPolicy>(rowrange, [=] (int r) {
    for(int c : colrange) {
      // TODO: indirect does not work here in clang
      maxloc_reducer.maxloc(actualdata[r * 10 + c], Index2D(c, r));
    }
  });

  double raja_max = (double)maxloc_reducer.get();
  Index2D raja_loc = maxloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxlocx, raja_loc.idx);
  ASSERT_EQ(this->maxlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionTupleLocTestTargetOMP, ReduceMaxLocTuple)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  RAJA::ReduceMaxLoc<RAJA::omp_target_reduce, double, RAJA::tuple<int, int>> maxloc_reducer(-1024.0, RAJA::make_tuple(0, 0));

  auto actualdata = this->data;
  auto indirect = this->array;
  RAJA::forall<ExecPolicy>(rowrange, [=] (int r) {
    for(int c : colrange) {
      // TODO: indirect does not work here in clang
      maxloc_reducer.maxloc(actualdata[r * 10 + c], RAJA::make_tuple(c, r));
    }
  });

  double raja_max = (double)maxloc_reducer.get();
  RAJA::tuple<int,int> raja_loc = maxloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxlocx, RAJA::get<0>(raja_loc));
  ASSERT_EQ(this->maxlocy, RAJA::get<1>(raja_loc));
}

REGISTER_TYPED_TEST_SUITE_P(ReductionTupleLocTestTargetOMP,
                            ReduceMaxLocIndex,
                            ReduceMaxLocTuple );

// TODO: Complete parameterization later when Clang runs properly.
// For now, un-parameterized for clarity and debugging.
using types =
    ::testing::Types<std::tuple<RAJA::omp_target_parallel_for_exec<16>,
                                RAJA::ReduceMinLoc<RAJA::omp_target_reduce, double, RAJA::tuple<int, int>>>,
                     std::tuple<RAJA::omp_target_parallel_for_exec<64>,
                                RAJA::ReduceMinLoc<RAJA::omp_target_reduce, double, RAJA::tuple<int, int>>>,
                     std::tuple<RAJA::omp_target_parallel_for_exec<256>,
                                RAJA::ReduceMinLoc<RAJA::omp_target_reduce, double, RAJA::tuple<int, int>>>
                               >;

INSTANTIATE_TYPED_TEST_SUITE_P(Reduce, ReductionTupleLocTestTargetOMP, types);


