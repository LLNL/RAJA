//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and
/// initialization.
///

#include "tests/test-reducer-constructors.hpp"

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPBasicReducerConstructorTypes = Test<camp::cartesian_product<
    OpenMPReducerPolicyList,
    DataTypeList,
    HostResourceList>>::Types;

using OpenMPInitReducerConstructorTypes = Test<camp::cartesian_product<
    OpenMPReducerPolicyList,
    DataTypeList,
    HostResourceList,
    SequentialUnitTestPolicyList>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(
    OpenMPBasicTest,
    ReducerBasicConstructorUnitTest,
    OpenMPBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(
    OpenMPInitTest,
    ReducerInitConstructorUnitTest,
    OpenMPInitReducerConstructorTypes);
#endif
