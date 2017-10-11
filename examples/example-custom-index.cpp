//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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
#include <algorithm>
#include <iostream>
#include <initializer_list>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/index/RangeSegment.hpp"

const int DIM = 2;

/*
  Example 5: Custom Index Set

  ----[Details]-------------------
  This example illustrates how to construct a custom
  iteration space composed of segments. Here a segment
  is an arbitrary collection of indices.

  Assuming a grid with the following contents

  grid = [1, 2, 1, 2,
          3, 4, 3, 4,
          1, 2, 1, 2,
          3, 4, 3, 4];

  The following code will construct four segments wherein
  each segment will store indices corresponding to a particular
  value on the grid. For example the first segment will store the
  indices {0,2,8,10} corresponding to the location of values equal to 1.

  --------[RAJA Concepts]---------
  1. Constructing custom IndexSets
  2. RAJA::View              - RAJA's wrapper for multidimensional indexing
  3. RAJA::ListSegment       - Container for an arbitrary collection of indices
  4. RAJA::TypedListSegment  - Container for an arbitrary collection of typed
  indices
  5. RAJA::StaticIndexSet    - Container for an index set which is a collection
  of
                               ListSegments
*/
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 5. Custom Index Set \n");
  int n = 4;
  int *A = new int[n * n];

  auto init = {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4};

  std::copy(init.begin(), init.end(), A);

  /*
    The template arguments for StaticIndexSet enables the user to indicate
    the required storage types of various segments. In this example,
    we only need to store TypedListSegment<Index_type> (aka ListSegment)
  */
  RAJA::StaticIndexSet<RAJA::TypedListSegment<RAJA::Index_type>> colorset;

  /*
    RAJA::View - RAJA's wrapper for multidimensional indexing
   */
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, n, n);

  /*
    Buffer used for intermediate indices storage
   */
  auto *idx = new RAJA::Index_type[(n + 1) * (n + 1) / 4];

  /*
    Iterate over each dimension (DIM=2 for this example)
  */
  for (int xdim : {0, 1}) {
    for (int ydim : {0, 1}) {

      RAJA::Index_type count = 0;

      /*
        Iterate over each extent in each dimension, incrementing by two to
        safely advance over neighbors
       */
      for (int xiter = xdim; xiter < n; xiter += 2) {
        for (int yiter = ydim; yiter < n; yiter += 2) {

          /*
            Add the computed index to the buffer
          */
          idx[count] = std::distance(std::addressof(Aview(0, 0)),
                                     std::addressof(Aview(xiter, yiter)));
          ++count;
        }
      }

      /*
        RAJA::ListSegment - creates a list segment from a given array with a
        specific length.

        Here the indicies are inserted from the buffer as a new ListSegment.
      */
      colorset.push_back(RAJA::ListSegment(idx, count));
    }
  }

  delete[] idx;


/*
  -----[RAJA Loop Traversal]-------
  Under the custom color policy, a RAJA forall loop will transverse
  through each list segment stored in the colorset sequentially and transverse
  each segment in parallel (if enabled).
 */
#if defined(RAJA_ENABLE_OPENMP)
  using ColorPolicy =
      RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>;
#else
  using ColorPolicy = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
#endif

  RAJA::forall<ColorPolicy>(
   colorset, [=](int idx) {
   
     printf("A[%d] = %d\n", idx, A[idx]);

   });

  return 0;
}
