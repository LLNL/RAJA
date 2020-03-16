//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-scan-inclusive.hpp"

#if defined(RAJA_ENABLE_TBB)

// TBB policy types to test
using TBBExecTypes = list< RAJA::tbb_for_exec, 
                           RAJA::tbb_for_dynamic >;

using TBBInclusiveScanTypes = 
  Test<cartesian_product< TBBExecTypes, ListHostRes, OpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB, 
                               ScanFunctionalTest, 
                               TBBInclusiveScanTypes);

#endif
