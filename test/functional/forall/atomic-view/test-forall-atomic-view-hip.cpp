//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for atomic operations with forall and views.
///

#include "tests/test-forall-atomic-view.hpp"

#include "../test-forall-execpol.hpp"

#include "../test-forall-atomic-utils.hpp"

#if defined(RAJA_ENABLE_HIP)
using HipAtomicForallViewTypes = Test< camp::cartesian_product<
                                                                HipForallExecPols,
                                                                HipAtomicPols,
                                                                HipResourceList,
                                                                AtomicDataTypeList >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( HipTest,
                                ForallAtomicViewFunctionalTest,
                                HipAtomicForallViewTypes );
#endif
