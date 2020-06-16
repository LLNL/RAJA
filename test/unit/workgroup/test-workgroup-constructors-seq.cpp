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

using SequentialBasicWorkGroupConstructorTypes =
  Test< camp::cartesian_product< SequentialExecPolicyList,
                                 SequentialOrderPolicyList,
                                 SequentialStoragePolicyList,
                                 IndexTypeTypeList,
                                 XargsTypeList,
                                 HostAllocatorList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(SequentialBasicTest,
                               WorkGroupBasicConstructorUnitTest,
                               SequentialBasicWorkGroupConstructorTypes);
