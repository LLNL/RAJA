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

#if defined(RAJA_ENABLE_HIP)
using AtomicHipExecs = list< RAJA::hip_exec<256> >;

using AtomicHipPols = list< RAJA::auto_atomic,
                            RAJA::hip_atomic
                          >;

using HipAtomicForallBasicTypes = Test< cartesian_product<
                                          AtomicHipExecs,
                                          AtomicHipPols,
                                          HipResourceList,
                                          AtomicDataTypes >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( HipTest,
                                HipForallAtomicBasicFunctionalTest,
                                HipAtomicForallBasicTypes );
#endif
