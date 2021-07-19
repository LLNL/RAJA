//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 * EXERCISE #6: Nested Loop Reordering
 *
 *  In this exercise, you will use RAJA::kernel execution policies 
 *  to permute the order of loops in a triple loop nest. In particular,
 *  you will reorder loop statements in execution policies. The exercise
 *  does no actual computation and just prints out the loop indices to show 
 *  the different orderings.
 *
 *  To avoid the complexity of interpreting parallel output, the execution
 *  policies you will write will use sequential execution.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - 'RAJA::kernel' loop abstractions and execution policies
 *    - Nested loop reordering
 *    - Strongly-typed loop indices
 */

//
// Define three named loop index types used in the triply-nested loops.
// These will trigger compilation errors if lambda index argument ordering 
// and types do not match the typed range index ordering.  See final
// example in this file.
//
RAJA_INDEX_VALUE(KIDX, "KIDX");
RAJA_INDEX_VALUE(JIDX, "JIDX"); 
RAJA_INDEX_VALUE(IIDX, "IIDX"); 


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #7: RAJA nested loop reorder example...\n";

  std::cout << "\n Running C-style loop nest with loop ordering: K-outer, J-middle, I-inner" 
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  for (int k = 2; k < 4; ++k) {
    for (int j = 1; j < 3; ++j) {
      for (int i = 0; i < 2; ++i) {
        printf( " (%d, %d, %d) \n", i, j, k);
      }
    }
  }

//
// The RAJA variants of the loop nest used following typed range segments
// based on the typed indices defined above, outside of main().
//
  RAJA::TypedRangeSegment<KIDX> KRange(2, 4);
  RAJA::TypedRangeSegment<JIDX> JRange(1, 3);
  RAJA::TypedRangeSegment<IIDX> IRange(0, 2);
 
//----------------------------------------------------------------------------//
 
  std::cout << "\n\n Running RAJA nested loop example (K-outer, J-middle, I-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  using KJI_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<2, RAJA::seq_exec,    // k
                          RAJA::statement::For<1, RAJA::seq_exec,  // j
                            RAJA::statement::For<0, RAJA::seq_exec,// i 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<KJI_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) { 
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });


//----------------------------------------------------------------------------//
 
  std::cout << "\n Running RAJA nested loop example (J-outer, I-middle, K-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  using JIK_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<1, RAJA::seq_exec,    // j
                          RAJA::statement::For<0, RAJA::seq_exec,  // i
                            RAJA::statement::For<2, RAJA::seq_exec,// k 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<JIK_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) { 
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });


//----------------------------------------------------------------------------//
 
  std::cout << "\n Running RAJA nested loop example (I-outer, K-middle, J-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  using IKJ_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<0, RAJA::seq_exec,    // i
                          RAJA::statement::For<2, RAJA::seq_exec,  // k
                            RAJA::statement::For<1, RAJA::seq_exec,// j 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;

  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) {
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });


#if 0
//----------------------------------------------------------------------------//
// The following demonstrates that code will not compile if lambda argument
// types/order do not match the types/order For statements in the execution
// policy. To see this, enable this code section and try to compile this file.
//----------------------------------------------------------------------------//

  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (JIDX i, IIDX j, KIDX k) {
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });

#endif

  std::cout << "\n DONE!...\n";

  return 0;
}

