//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and initialization.
///

#include "tests/test-reducer-constructors.hpp"

#if defined(RAJA_ENABLE_TBB)
using TBBBasicReducerConstructorTypes = 
  Test< camp::cartesian_product< TBBReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList > >::Types;

using TBBInitReducerConstructorTypes = 
  Test< camp::cartesian_product< TBBReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList,
                                 SequentialForoneList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBBBasicTest,
                               ReducerBasicConstructorUnitTest,
                               TBBBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(TBBInitTest,
                               ReducerInitConstructorUnitTest,
                               TBBInitReducerConstructorTypes);
#endif

