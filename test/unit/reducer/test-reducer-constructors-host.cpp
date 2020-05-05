//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and initialization.
///

#include "tests/test-reducer-constructors.hpp"

#include "test-reducer-utils.hpp"

using HostBasicReducerConstructorTypes = Test< camp::cartesian_product<
                                                        HostReducerPolicyList,
                                                        DataTypeList,
                                                        HostResourceList
                                                      >
                             >::Types;

using HostInitReducerConstructorTypes = Test< camp::cartesian_product<
                                                        HostReducerPolicyList,
                                                        DataTypeList,
                                                        HostResourceList
                                                     >
                            >::Types;

INSTANTIATE_TYPED_TEST_CASE_P(HostBasicTest,
                              ReducerBasicConstructorUnitTest,
                              HostBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_CASE_P(HostInitTest,
                              ReducerInitConstructorUnitTest,
                              HostInitReducerConstructorTypes);


