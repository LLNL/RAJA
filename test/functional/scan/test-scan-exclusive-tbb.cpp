
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-scan-exclusive.hpp"

#if defined(RAJA_ENABLE_TBB)

// TBB policy types to test
using TBBExecTypes = camp::list< RAJA::tbb_for_exec,
                                 RAJA::tbb_for_static< >,
                                 RAJA::tbb_for_static< 2 >,
                                 RAJA::tbb_for_static< 4 >,
                                 RAJA::tbb_for_static< 8 >,
                                 RAJA::tbb_for_dynamic >;

using TBBExclusiveScanTypes = 
  Test<cartesian_product< TBBExecTypes, ListHostRes, OpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB, 
                               ScanFunctionalTest, 
                               TBBExclusiveScanTypes);

#endif
