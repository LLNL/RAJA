//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA util intro_sort for hip gpus
///

#include "test-util-sort.hpp"

#if defined(RAJA_ENABLE_HIP)

using HipIntroSortTypes = Test< camp::cartesian_product<
                                                             HipIntroSortSorters,
                                                             HipResourceList,
                                                             SortKeyTypeList,
                                                             SortMaxNListSmall >
                         >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( HipTest,
                                SortUnitTest,
                                HipIntroSortTypes );

#endif

