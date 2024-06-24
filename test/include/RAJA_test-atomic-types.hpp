//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Type list for testing RAJA atomics.
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
              unsigned int,
              long long,
              unsigned long long,
              float,
              double >;

#endif // __RAJA_test_atomic_types_HPP__
