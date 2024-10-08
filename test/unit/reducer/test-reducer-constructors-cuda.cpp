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

#if defined(RAJA_ENABLE_CUDA)
using CudaBasicReducerConstructorTypes = 
  Test< camp::cartesian_product< CudaReducerPolicyList,
                                 DataTypeList,
                                 CudaResourceList > >::Types;

using CudaInitReducerConstructorTypes = 
  Test< camp::cartesian_product< CudaReducerPolicyList,
                                 DataTypeList,
                                 CudaResourceList,
                                 CudaUnitTestPolicyList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(CudaBasicTest,
                               ReducerBasicConstructorUnitTest,
                               CudaBasicReducerConstructorTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(CudaInitTest,
                               ReducerInitConstructorUnitTest,
                               CudaInitReducerConstructorTypes);
#endif

