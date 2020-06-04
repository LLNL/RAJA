//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for atomic operations with forall.
///

#include "tests/test-forall-atomic-basic.hpp"

#include "../test-forall-execpol.hpp"

#include "../test-forall-atomic-utils.hpp"

#if defined(RAJA_ENABLE_HIP)
using HipAtomicForallBasicTypes = Test< camp::cartesian_product<
                                                                 HipForallExecPols,
                                                                 HipAtomicPols,
                                                                 HipResourceList,
                                                                 AtomicSegmentList,
                                                                 AtomicDataTypeList >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( HipTest,
                                ForallAtomicBasicFunctionalTest,
                                HipAtomicForallBasicTypes );
#endif
