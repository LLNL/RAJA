//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup constructors and initialization.
///

#include "tests/test-workgroup-constructors.hpp"

#if defined(RAJA_ENABLE_TBB)
using TBBBasicWorkGroupConstructorTypes =
  Test< camp::cartesian_product< TBBExecPolicyList,
                                 TBBOrderPolicyList,
                                 TBBStoragePolicyList,
                                 IndexTypeTypeList,
                                 XargsTypeList,
                                 HostAllocatorList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBBBasicTest,
                               WorkGroupBasicConstructorUnitTest,
                               TBBBasicWorkGroupConstructorTypes);
#endif
