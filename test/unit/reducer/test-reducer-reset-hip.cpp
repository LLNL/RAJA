//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer reset.
///

#include "tests/test-reducer-reset.hpp"

#if defined(RAJA_ENABLE_HIP)
using HipReducerResetTypes = 
  Test< camp::cartesian_product< HipReducerPolicyList,
                                 DataTypeList,
                                 HipResourceList,
                                 HipUnitTestPolicyList > >::Types;


INSTANTIATE_TYPED_TEST_SUITE_P(HipResetTest,
                               ReducerResetUnitTest,
                               HipReducerResetTypes);
#endif
