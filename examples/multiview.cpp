//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include <cstdio>
#include <array>

/*
 * MultiView Usage Example
 *
 * RAJA::MultiView is a view on an array-of-pointers
 * (or pointer-to-pointers), rather than a RAJA::View on a single
 * array. This allows a single RAJA::Layout to be applied to a
 * large set of arrays, and encapsulates a set of arrays more
 * easily.
 *
 * Instantiation of a MultiView works exactly like a standard View:
 *
 * RAJA::MultiView<data_type, RAJA::Layout<dim,index_type>> myMView(arrayofpointers, layout)
 *
 * Accessing the MultiView is similar to View, with an additional
 * array-of-pointers index in the 0th position. The following example
 * obtains the (x,y)th element of the 3rd array in an array-of-pointers:
 *
 * myMView(3, x, y);
 *
 * Optionally, the position of the array-of-pointers index can be
 * "swizzled" within the MultiView access. This is an example of the
 * same previous access, with the array-of-pointers index swizzled
 * to the 2nd position (the default position used in the previous 
 * example is the 0th position):
 *
 * RAJA::MultiView<data_type, RAJA::Layout<dim,index_type>, 2> myMView(arrayofpointers, layout)
 * myMView(x, y, 3);
 *
 * The following code demonstrates 2 aspects of RAJA::MultiView usage:
 * - Basic usage
 * - Swizzling of the array-of-pointers index
 */

int main()
{
  constexpr int N = 12;
  int * myarr[2]; // two 3x4 arrays
  int arr1[N];
  int arr2[N];

  for ( int ii = 0; ii < N; ++ii )
  {
    arr1[ii] = 100 + ii;
    arr2[ii] = 200 + ii;
  }

  myarr[0] = arr1;
  myarr[1] = arr2;

  // 4x3 layout
  std::array<RAJA::idx_t, 2> perm { {0, 1} };
  RAJA::Layout<2> layout = RAJA::make_permuted_layout(
                              { {4, 3} }, perm
                           );

  // Basic MultiView usage
  // Default usage: no specified array-of-pointers index swizzling
  // 0th position is used as the array-of-pointers index
  RAJA::MultiView<int, RAJA::Layout<2, RAJA::Index_type>> arrView(myarr, layout);

  // Swizzled MultiView usage
  // Add an array-of-pointers index swizzle specifier
  constexpr int swizzle = 1;
  RAJA::MultiView<int, RAJA::Layout<2, RAJA::Index_type>, swizzle> arrViewSwizz(myarr, layout);

  // Comparing values of both views
  printf ( "Comparing values of both default and 1-index-swizzled MultiViews:\n" );
  for ( int pp = 0; pp < 2; ++pp )
  {
    for ( int kk = 0; kk < 4; ++kk )
    {
      for ( int jj = 0; jj < 3; ++jj )
      {
        printf ( "arr(%i, %i, %i) %d == arrswizz(%i, %i, %i) %d\n", pp, kk, jj, arrView(pp, kk, jj), kk, pp, jj, arrViewSwizz(kk, pp, jj) );
      }
    }
  }

  // switch values
  printf ( "Switching values\n" );
  for ( int kk = 0; kk < 4; ++kk )
  {
    for ( int jj = 0; jj < 3; ++jj )
    {
      int temp = arrView(0, kk, jj);
      arrView(0, kk, jj) = arrView(1, kk, jj);
      arrView(1, kk, jj) = temp;
    }
  }

  // Comparing switched values of both views
  printf ( "Comparing switched values of both default and 1-index-swizzled MultiViews:\n" );
  for ( int pp = 0; pp < 2; ++pp )
  {
    for ( int kk = 0; kk < 4; ++kk )
    {
      for ( int jj = 0; jj < 3; ++jj )
      {
        printf ( "arr(%i, %i, %i) %d == arrswizz(%i, %i, %i) %d\n", pp, kk, jj, arrView(pp, kk, jj), kk, pp, jj, arrViewSwizz(kk, pp, jj) );
      }
    }
  }

  return 0;
}
