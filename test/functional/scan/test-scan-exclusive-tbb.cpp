
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-scan-exclusive.hpp"

#if defined(RAJA_ENABLE_TBB)

#include "../forall/test-forall-utils.hpp"
#include "../forall/test-forall-execpol.hpp"

using TBBExclusiveScanTypes = 
  Test<camp::cartesian_product< TBBForallExecPols, 
                                HostResourceList, 
                                ScanOpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB, 
                               ScanFunctionalTest, 
                               TBBExclusiveScanTypes);

#endif
