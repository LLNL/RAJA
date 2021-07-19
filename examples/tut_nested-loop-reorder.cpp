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
 *  Nested Loop Reorder Example
 *
 *  This example shows how to reorder RAJA nested loops by reordering
 *  nested policy arguments. It does no actual computation and just
 *  prints out the loop indices to show the different orderings.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - 'RAJA::kernel' loop abstractions and execution policies
 *    - Nested loop reordering
 *    - Strongly-typed loop indices
 */

//
// Define three named loop index types used in the triply-nested loop examples.
// These will trigger compilation errors if lambda index argument ordering 
// and types do not match the typed range index ordering.
//
// _nestedreorder_idxtypes_start
RAJA_INDEX_VALUE(KIDX, "KIDX");
RAJA_INDEX_VALUE(JIDX, "JIDX"); 
RAJA_INDEX_VALUE(IIDX, "IIDX"); 
// _nestedreorder_idxtypes_end


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

    std::cout << "\n\nRAJA nested loop reorder example...\n";

//
// Typed index ranges
// 
// _nestedreorder_ranges_start
  RAJA::TypedRangeSegment<KIDX> KRange(2, 4);
  RAJA::TypedRangeSegment<JIDX> JRange(1, 3);
  RAJA::TypedRangeSegment<IIDX> IRange(0, 2);
// _nestedreorder_ranges_end
 
//----------------------------------------------------------------------------//
 
  std::cout << "\n Running loop reorder example (K-outer, J-middle, I-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _nestedreorder_kji_start
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
  // _nestedreorder_kji_end


//----------------------------------------------------------------------------//
 
  std::cout << "\n Running loop reorder example (J-outer, I-middle, K-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _nestedreorder_jik_start
  using JIK_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<1, RAJA::seq_exec,    // j
                          RAJA::statement::For<0, RAJA::seq_exec,  // i
                            RAJA::statement::For<2, RAJA::seq_exec,// k 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;
  // _nestedreorder_jik_end

  RAJA::kernel<JIK_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) { 
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });


//----------------------------------------------------------------------------//
 
  std::cout << "\n Running loop reorder example (I-outer, K-middle, J-inner)"
            << "...\n\n" << " (I, J, K)\n" << " ---------\n";

  // _nestedreorder_ikj_start
  using IKJ_EXECPOL = RAJA::KernelPolicy<
                        RAJA::statement::For<0, RAJA::seq_exec,    // i
                          RAJA::statement::For<2, RAJA::seq_exec,  // k
                            RAJA::statement::For<1, RAJA::seq_exec,// j 
                              RAJA::statement::Lambda<0>
                            > 
                          > 
                        > 
                      >;
  // _nestedreorder_ikj_end

  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (IIDX i, JIDX j, KIDX k) {
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });


#if 0
//----------------------------------------------------------------------------//
// The following demonstrates that code will not compile if lambda argument
// types/order do not match the types/order of the For statements.
//----------------------------------------------------------------------------//

  // _nestedreorder_typemismatch_start
  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
  [=] (JIDX i, IIDX j, KIDX k) {
     printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
  });
  // _nestedreorder_typemismatch_end

#endif

  std::cout << "\n DONE!...\n";

  return 0;
}

