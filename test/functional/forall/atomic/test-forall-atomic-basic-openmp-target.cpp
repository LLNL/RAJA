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

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OmpTargetAtomicForallBasicTypes = Test< camp::cartesian_product<
                                                                       OpenMPTargetForallExecPols,
                                                                       OpenMPAtomicPols,
                                                                       OpenMPTargetResourceList,
                                                                       AtomicSegmentList,
                                                                       AtomicDataTypeList >
                                            >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OmpTargetTest,
                                ForallAtomicBasicFunctionalTest,
                                OmpTargetAtomicForallBasicTypes );
#endif
