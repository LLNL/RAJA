//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-rangesegment.hpp"

// Generate TBB Type List
#if defined(RAJA_ENABLE_TBB)
using TBBTypes = list< RAJA::tbb_for_exec,
                       RAJA::tbb_for_static<8>,
                       RAJA::tbb_for_dynamic
                     >;

using TBBForallTypes = Test<cartesian_product< IdxTypes, ListHost, TBBTypes>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB, ForallFunctionalTest, TBBForallTypes);
#endif
