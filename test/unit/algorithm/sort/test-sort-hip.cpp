//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with hip policies
///

#include "test-sort.hpp"

#if defined(RAJA_ENABLE_HIP)

using HipSortTypes = Test< camp::cartesian_product<
                                                    HipSortSorters,
                                                    HipResourceList,
                                                    SortKeyTypeList,
                                                    SortMaxNListDefault >
                         >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( HipTest,
                                SortUnitTest,
                                HipSortTypes );

#endif

