//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup ordered runs.
///

#include "tests/test-workgroup-ordered.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetBasicWorkGroupOrderedTypes =
  Test< camp::cartesian_product< OpenMPTargetExecPolicyList,
                                 OpenMPTargetOrderedPolicyList,
                                 OpenMPTargetStoragePolicyList,
                                 IndexTypeTypeList,
                                 OpenMPTargetAllocatorList,
                                 OpenMPTargetResourceList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPTargetBasicTest,
                               WorkGroupBasicOrderedFunctionalTest,
                               OpenMPTargetBasicWorkGroupOrderedTypes);
#endif
