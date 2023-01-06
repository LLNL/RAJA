//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Custom index type used for loc reductions.
//

#ifndef __RAJA_test_reduceloc_types_HPP__
#define __RAJA_test_reduceloc_types_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

struct Index2D {
   RAJA::Index_type idx, idy;
   constexpr Index2D() : idx(-1), idy(-1) {}
   constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
};

#endif // __RAJA_test_reduceloc_types_HPP__
