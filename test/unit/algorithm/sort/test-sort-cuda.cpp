//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with cuda policies
///

#include "test-sort.hpp"

#if defined(RAJA_ENABLE_CUDA)

using CudaSortTypes = Test< camp::cartesian_product<
                                                     CudaSortSorters,
                                                     CudaResourceList,
                                                     SortKeyTypeList,
                                                     SortMaxNListDefault >
                          >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( CudaTest,
                                SortUnitTest,
                                CudaSortTypes );

#endif

