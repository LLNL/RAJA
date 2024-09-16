//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Types and type lists for reductions used throughout RAJA tests.
//
// Note that in the type lists, a subset of types is used by default.
// For more comprehensive type testing define the macro RAJA_TEST_EXHAUSTIVE.
//

#ifndef __RAJA_test_reduce_types_HPP__
#define __RAJA_test_reduce_types_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//
// Reduce data types
//
using ReduceDataTypeList = camp::list<
    int,
#if defined(RAJA_TEST_EXHAUSTIVE)
    unsigned,
    long long,
    unsigned long long,
#endif
    float,
    double>;

#endif  // __RAJA_test_reduce_types_HPP__
