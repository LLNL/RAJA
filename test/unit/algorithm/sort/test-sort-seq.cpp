//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with sequential policies
///

#include "test-sort.hpp"

using SeqSortTypes = Test< camp::cartesian_product<
                                                    SeqSortSorters,
                                                    HostResourceList,
                                                    SortKeyTypeList,
                                                    SortMaxNListDefault >
                         >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( SeqTest,
                                SortUnitTest,
                                SeqSortTypes );

