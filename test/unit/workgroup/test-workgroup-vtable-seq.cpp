//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup vtable.
///

#include "tests/test-workgroup-vtable.hpp"

using SequentialBasicWorkGroupVtableTypes =
  Test< camp::cartesian_product< SequentialExecPolicyList,
                                 IndexTypeTypeList,
                                 XargsTypeList,
                                 HostResourceList,
                                 SequentialForoneList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(SequentialBasicTest,
                               WorkGroupBasicVtableUnitTest,
                               SequentialBasicWorkGroupVtableTypes);
