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

using AtomicSeqExecs = list< RAJA::seq_exec >;

using AtomicSeqPols = list< RAJA::auto_atomic,
                            RAJA::seq_atomic,
                            RAJA::builtin_atomic
                          >;

using SeqAtomicForallBasicTypes = Test< cartesian_product<
                                          AtomicSeqExecs,
                                          AtomicSeqPols,
                                          HostResourceList,
                                          AtomicDataTypes >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( SeqTest,
                                SeqForallAtomicBasicFunctionalTest,
                                SeqAtomicForallBasicTypes );
