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

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPBasicWorkGroupUnorderedTypes =
  Test< camp::cartesian_product< OpenMPExecPolicyList,
                                 OpenMPOrderPolicyList,
                                 OpenMPStoragePolicyList,
                                 IndexTypeTypeList,
                                 HostAllocatorList,
                                 HostResourceList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPBasicTest,
                               WorkGroupBasicUnorderedFunctionalTest,
                               OpenMPBasicWorkGroupUnorderedTypes);
#endif
