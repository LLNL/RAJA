//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA util insertion_sort for cuda gpus
///

#include "test-util-sort.hpp"

#if defined(RAJA_ENABLE_CUDA)

using CudaInsertionSortTypes = Test< camp::cartesian_product<
                                                             CudaInsertionSortSorters,
                                                             CudaResourceList,
                                                             SortKeyTypeList,
                                                             SortMaxNListTiny >
                         >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( CudaTest,
                                SortUnitTest,
                                CudaInsertionSortTypes );

#endif

