//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup unordered runs.
///

#include "tests/test-workgroup-unordered.hpp"

#if defined(RAJA_ENABLE_TBB)
using TBBBasicWorkGroupUnorderedTypes =
  Test< camp::cartesian_product< TBBExecPolicyList,
                                 TBBOrderPolicyList,
                                 TBBStoragePolicyList,
                                 IndexTypeTypeList,
                                 HostAllocatorList,
                                 HostResourceList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBBBasicTest,
                               WorkGroupBasicUnorderedFunctionalTest,
                               TBBBasicWorkGroupUnorderedTypes);
#endif
