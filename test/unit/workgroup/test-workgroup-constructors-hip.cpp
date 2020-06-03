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

#include "test-workgroup-utils.hpp"

#if defined(RAJA_ENABLE_HIP)
using HipBasicWorkGroupConstructorTypes =
  Test< camp::cartesian_product< HipExecPolicyList,
                                 HipOrderPolicyList,
                                 HipStoragePolicyList,
                                 IndexTypeTypeList,
                                 XargsTypeList,
                                 HipAllocatorList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(HipBasicTest,
                               WorkGroupBasicConstructorUnitTest,
                               HipBasicWorkGroupConstructorTypes);
#endif

