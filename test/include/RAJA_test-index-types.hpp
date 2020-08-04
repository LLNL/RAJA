//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Types and type lists for loop indexing used throughout RAJA tests.
//
// Note that in the type lists, a subset of types is used by default.
// For more comprehensive type testing define the macro RAJA_TEST_EXHAUSTIVE.
//
// Also, some tests may define their own index types to test.
//

#ifndef __RAJA_test_index_types_HPP__
#define __RAJA_test_index_types_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//
// Strongly typed indexes
//
RAJA_INDEX_VALUE(StrongIndexType, "StrongIndexType");
RAJA_INDEX_VALUE_T(StrongInt, int, "StrongIntType");
RAJA_INDEX_VALUE_T(StrongULL, unsigned long long , "StrongULLType");

//
// Standard index types list
//
using IdxTypeList = camp::list<RAJA::Index_type,
                               int,
#if defined(RAJA_TEST_EXHAUSTIVE)
                               unsigned int,
                               short,
                               unsigned short,
                               long int,
                               unsigned long,
                               long long,
#endif
                               unsigned long long>;
//
// Index types w/ Strong types list
//
using StrongIdxTypeList = camp::list<RAJA::Index_type,
                                     int,
                                     StrongIndexType,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                     StrongInt,
                                     unsigned int,
                                     short,
                                     unsigned short,
                                     long int,
                                     unsigned long,
                                     long long,
#endif
                                     StrongULL,
                                     unsigned long long>;

#endif // __RAJA_test_index_types_HPP__
