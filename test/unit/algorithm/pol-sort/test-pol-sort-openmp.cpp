//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with openmp policies
///

#include "test-pol-sort.hpp"

#if defined(RAJA_ENABLE_OPENMP) && 0

using OpenmpSortTypes = Test< camp::cartesian_product<
                                                       OpenmpSortSorters,
                                                       HostResourceList,
                                                       SortKeyTypeList,
                                                       SortMaxNListDefault >
                            >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OpenmpTest,
                                SortUnitTest,
                                OpenmpSortTypes );

#endif

