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

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetBasicWorkGroupUnorderedTypes =
  Test< camp::cartesian_product< OpenMPTargetExecPolicyList,
                                 OpenMPTargetOrderPolicyList,
                                 OpenMPTargetStoragePolicyList,
                                 IndexTypeTypeList,
                                 OpenMPTargetAllocatorList,
                                 OpenMPTargetResourceList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTargetBasicTest,
                               WorkGroupBasicUnorderedFunctionalTest,
                               OpenMPTargetBasicWorkGroupUnorderedTypes);
#endif
