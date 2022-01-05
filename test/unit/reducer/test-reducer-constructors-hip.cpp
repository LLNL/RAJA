//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and initialization.
///

#include "tests/test-reducer-constructors.hpp"

#if defined(RAJA_ENABLE_HIP)
using HipInitReducerConstructorTypes = 
  Test< camp::cartesian_product< HipReducerPolicyList,
                                 DataTypeList,
                                 HipResourceList,
                                 HipForoneList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(HipInitTest,
                               ReducerInitConstructorUnitTest,
                               HipInitReducerConstructorTypes);
#endif

