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

#if defined(RAJA_ENABLE_HIP)
using HipBasicWorkGroupUnorderedTypes =
  Test< camp::cartesian_product< HipExecPolicyList,
                                 HipOrderPolicyList,
                                 HipStoragePolicyList,
                                 IndexTypeTypeList,
                                 HipAllocatorList,
                                 HipResourceList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(HipBasicTest,
                               WorkGroupBasicUnorderedFunctionalTest,
                               HipBasicWorkGroupUnorderedTypes);
#endif
