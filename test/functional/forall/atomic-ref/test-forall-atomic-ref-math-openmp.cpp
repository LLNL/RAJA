//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for arithmetic atomic operations
///

#include "tests/test-forall-atomic-ref-math.hpp"

#include "../test-forall-atomic-utils.hpp"

#if defined(RAJA_ENABLE_OPENMP)
using OmpAtomicForallRefMathTypes = Test< camp::cartesian_product<
                                                                  OpenMPForallAtomicExecPols,
                                                                  OpenMPAtomicPols,
                                                                  HostResourceList,
                                                                  AtomicDataTypeList >
                                        >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OmpTest,
                                ForallAtomicRefMathFunctionalTest,
                                OmpAtomicForallRefMathTypes );
#endif
