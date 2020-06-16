//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup workstorage.
///

#include "tests/test-workgroup-workstorage.hpp"

using BasicWorkGroupWorkStorageTypes =
  Test< camp::cartesian_product< SequentialStoragePolicyList,
                                 HostAllocatorList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(SequentialBasicTest,
                               WorkGroupBasicWorkStorageUnitTest,
                               BasicWorkGroupWorkStorageTypes);
