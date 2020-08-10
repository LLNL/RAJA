//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

TEST(MultiViewUnitTest, BasicTest)
{
  constexpr int N = 12;
  int * myarr[2];
  int arr1[N];
  int arr2[N];

  for ( int ii = 0; ii < 12; ++ii )
  {
    arr1[ii] = 100 + ii;
    arr2[ii] = 200 + ii;
  }

  myarr[0] = arr1;
  myarr[1] = arr2;

  // interleaved layout
  std::array<RAJA::idx_t, 2> perm { {1, 0} };
  RAJA::Layout<2> layout = RAJA::make_permuted_layout(
                              { {2, 6} }, perm
                           );

  // multi array of pointers view
  constexpr int swizzle = 1;
  RAJA::MultiView<int, RAJA::Layout<2, RAJA::Index_type, 0>, int **, swizzle> arrView(myarr, layout);

  for ( int zz = 0; zz < 2; ++zz )
  {
    for ( int kk = 0; kk < 2; ++kk )
    {
      for ( int jj = 0; jj < 6; ++jj )
      {
        printf ( "arr%i(%i, %i) %d\n", zz, kk, jj, arrView(zz, kk, jj) );
      }
    }
  }

  // switch values
  printf ( "Switching values\n" );
  for ( int kk = 0; kk < 2; ++kk )
  {
    for ( int jj = 0; jj < 6; ++jj )
    {
      int temp = arrView(0, kk, jj);
      arrView(0, kk, jj) = arrView(1, kk, jj);
      arrView(1, kk, jj) = temp;
    }
  }

  for ( int zz = 0; zz < 2; ++zz )
  {
    for ( int kk = 0; kk < 2; ++kk )
    {
      for ( int jj = 0; jj < 6; ++jj )
      {
        printf ( "arr%i(%i, %i) %d\n", zz, kk, jj, arrView(zz, kk, jj) );
      }
    }
  }
}

