//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment-view.hpp"

#if defined(RAJA_ENABLE_TBB)

// TBB execution policy types
using TBBForallExecPols = camp::list< RAJA::tbb_for_exec,
                                      RAJA::tbb_for_static< >,
                                      RAJA::tbb_for_static< 2 >,
                                      RAJA::tbb_for_static< 4 >,
                                      RAJA::tbb_for_static< 8 >,
                                      RAJA::tbb_for_dynamic >;

// Cartesian product of types for TBB tests
using TBBForallSegmentTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                TBBForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB,
                               ForallSegmentViewTest,
                               TBBForallSegmentTypes);
#endif
