//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and initialization.
///

#include "tests/test-reducer-constructors.hpp"

using SequentialBasicReducerConstructorTypes = 
  Test< camp::cartesian_product< SequentialReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList > >::Types;

using SequentialInitReducerConstructorTypes = 
  Test< camp::cartesian_product< SequentialReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList,
                                 SequentialUnitTestPolicyList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(SequentialBasicTest,
                               ReducerBasicConstructorUnitTest,
                               SequentialBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(SequentialInitTest,
                               ReducerInitConstructorUnitTest,
                               SequentialInitReducerConstructorTypes);


