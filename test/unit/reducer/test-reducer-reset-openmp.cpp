//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer reset.
///

#include "tests/test-reducer-reset.hpp"

#include "test-reducer-utils.hpp"

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPReducerResetTypes = 
  Test< camp::cartesian_product< OpenMPReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList,
                                 SequentialForoneList > >::Types;


INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPResetTest,
                               ReducerResetUnitTest,
                               OpenMPReducerResetTypes);
#endif
