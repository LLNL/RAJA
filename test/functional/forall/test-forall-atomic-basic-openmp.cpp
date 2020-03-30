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

#if defined(RAJA_ENABLE_OPENMP)
using AtomicOmpExecs = list< RAJA::omp_for_exec,
                             RAJA::omp_for_nowait_exec,
                             RAJA::omp_parallel_for_exec
                           >;

using AtomicOmpPols = list< RAJA::auto_atomic,
                            RAJA::omp_atomic,
                            RAJA::builtin_atomic
                          >;

using OmpAtomicForallBasicTypes = Test< cartesian_product<
                                          AtomicOmpExecs,
                                          AtomicOmpPols,
                                          HostResourceList,
                                          AtomicDataTypes >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OmpTest,
                                SeqForallAtomicBasicFunctionalTest,
                                OmpAtomicForallBasicTypes );
#endif
