//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA stable sort with tbb policies
///

#include "test-stable-sort.hpp"

#if defined(RAJA_ENABLE_TBB) && 0

using TbbStableSortTypes = Test< camp::cartesian_product<
                                                          TbbStableSortSorters,
                                                          HostResourceList,
                                                          SortKeyTypeList,
                                                          SortMaxNListDefault >
                               >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( TbbTest,
                                SortUnitTest,
                                TbbStableSortTypes );

#endif

