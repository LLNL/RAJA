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

#if defined(RAJA_ENABLE_TBB)
using TBBReducerResetTypes = 
  Test< camp::cartesian_product< TBBReducerPolicyList,
                                 DataTypeList,
                                 HostResourceList,
                                 SequentialForoneList > >::Types;


INSTANTIATE_TYPED_TEST_SUITE_P(TBBResetTest,
                               ReducerResetUnitTest,
                               TBBReducerResetTypes);
#endif
