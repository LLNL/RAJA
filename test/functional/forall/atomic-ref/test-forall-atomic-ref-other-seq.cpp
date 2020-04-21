//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for logical, accessor, min/max, and cas atomic operations
///

#include "tests/test-forall-atomic-ref-other.hpp"

#include "../test-forall-atomic-utils.hpp"

using SeqAtomicForallRefOtherTypes = Test< camp::cartesian_product<
                                                                  AtomicSeqExecs,
                                                                  AtomicSeqPols,
                                                                  HostResourceList,
                                                                  AtomicTypeList>
                                        >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( SeqTest,
                                SeqForallAtomicRefOtherFunctionalTest,
                                SeqAtomicForallRefOtherTypes );
