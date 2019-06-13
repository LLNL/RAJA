//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
/// Source file containing tests for RAJA CPU reduction deaths.
///

#include "gtest/gtest.h"

#include <iostream>
#include "RAJA/RAJA.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"

// EXPERIMENTAL: Trying gtest death test via Travis. Will add more tests if successful.

/* For parameterization if death test works.
template <typename T>
struct BasicReduceLocTest : public ::testing::Test
{
  public:
  virtual void SetUp()
  {
    // 2 dimensional, 10x10
    array_length = 100;
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

  virtual void TearDown() {
    RAJA::free_aligned(array);
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

TYPED_TEST_CASE_P ( BasicReduceLocTest );

using BasicReduceLocDeathTest = BasicReduceLocTest;
*/


constexpr int array_length = 100;

TEST( BasicReduceLocDeathTest, OutOfBounds )
{
  RAJA::Real_ptr * array = RAJA::allocate_aligned_type<double *>(RAJA::DATA_ALIGN,
                                                  ydim * sizeof(double *));
  RAJA::Real_ptr data = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                               array_length * sizeof(double));

  RAJA::ReduceMinLoc<RAJA::seq_reduce, int, RAJA::tuple<int, int>> reduce_minloc(0, RAJA::make_tuple(0, 0));

  EXPECT_DEATH_IF_SUPPORTED( reduce_minloc.minloc( array[0][0], RAJA::make_tuple(10, 10) ), "" ); // not supported at runtime
  //ASSERT_DEATH_IF_SUPPORTED( reduce_minloc.minloc( array[0][0], RAJA::make_tuple(10, 10) ), "" ); // not supported at runtime
  //EXPECT_EXIT( reduce_minloc.minloc( array[0][0], RAJA::make_tuple(10, 10) ), ::testing::KilledBySignal(SIGSEGV), "ReduceMinLoc tuple out of bounds access." ); // does not compile
}

