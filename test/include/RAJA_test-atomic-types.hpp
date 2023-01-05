//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Types and type lists for loop indexing used throughout RAJA tests.
//
// Note that in the type lists, a subset of types is used by default.
// For more comprehensive type testing define the macro RAJA_TEST_EXHAUSTIVE.
//

#ifndef __RAJA_test_atomic_types_HPP__
#define __RAJA_test_atomic_types_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//
// Atomic data types
//
using AtomicDataTypeList =
  camp::list< RAJA::Index_type,
              int,
#if defined(RAJA_TEST_EXHAUSTIVE)
              unsigned,
              long long,
              unsigned long long,
              float,
#endif
              double >;

using AtomicDataUnsignedTypeList =
  camp::list< unsigned,
#if defined(RAJA_TEST_EXHAUSTIVE)
              unsigned long,
#endif
              unsigned long long>;

#endif // __RAJA_test_atomic_types_HPP__
