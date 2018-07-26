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
 *    - 'RAJA::nested' loop abstractions and execution policies
 *    - Nested loop reordering
 *    - Strongly-typed loop indices
 */

//
// Define three named loop index types used in the triply-nested loop examples.
// These will trigger compilation errors if lambda index argument ordering 
// and types do not match the typed range index ordering.
//
RAJA_INDEX_VALUE(KIDX, "KIDX");
RAJA_INDEX_VALUE(JIDX, "JIDX"); 
RAJA_INDEX_VALUE(IIDX, "IIDX"); 


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

    std::cout << "\n\nRAJA nested loop reorder example...\n";

//
// Typed index ranges
// 
  RAJA::TypedRangeSegment<KIDX> KRange(2, 4);
  RAJA::TypedRangeSegment<JIDX> JRange(1, 3);
  RAJA::TypedRangeSegment<IIDX> IRange(0, 2);
 
//----------------------------------------------------------------------------//
 
  std::cout << "\n Running loop reorder example (K-outer, J-middle, I-inner)"
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
 
  std::cout << "\n Running loop reorder example (J-outer, I-middle, K-inner)"
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
 
  std::cout << "\n Running loop reorder example (I-outer, K-middle, J-inner)"
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
// types/order do not match the types/order of the strongly-typed nested::For.
//----------------------------------------------------------------------------//

  RAJA::kernel<IKJ_EXECPOL>( RAJA::make_tuple(IRange, JRange, KRange),
    [=] (JIDX i, IIDX j, KIDX k) {
       printf( " (%d, %d, %d) \n", (int)(*i), (int)(*j), (int)(*k));
    });

#endif

  std::cout << "\n DONE!...\n";

  return 0;
}

