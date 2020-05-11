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

#if defined(RAJA_ENABLE_TBB)
using TbbBasicReducerConstructorTypes = Test< camp::cartesian_product<
                                                        TbbReducerPolicyList,
                                                        DataTypeList,
                                                        HostResourceList
                                                      >
                             >::Types;

using TbbInitReducerConstructorTypes = Test< camp::cartesian_product<
                                                        TbbReducerPolicyList,
                                                        DataTypeList,
                                                        HostResourceList,
                                                        SequentialForoneList
                                                     >
                            >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TbbBasicTest,
                               ReducerBasicConstructorUnitTest,
                               TbbBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(TbbInitTest,
                               ReducerInitConstructorUnitTest,
                               TbbInitReducerConstructorTypes);
#endif

