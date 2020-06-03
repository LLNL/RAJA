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

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPBasicWorkGroupConstructorTypes =
  Test< camp::cartesian_product< OpenMPExecPolicyList,
                                 OpenMPOrderPolicyList,
                                 OpenMPStoragePolicyList,
                                 IndexTypeTypeList,
                                 XargsTypeList,
                                 HostAllocatorList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPBasicTest,
                               WorkGroupBasicConstructorUnitTest,
                               OpenMPBasicWorkGroupConstructorTypes);
#endif

