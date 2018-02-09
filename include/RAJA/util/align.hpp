/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing an implementation of std align.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ALIGN_HPP
#define RAJA_ALIGN_HPP

#include "RAJA/config.hpp"

namespace RAJA
{

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Taken from libc++
// See libc++ license in docs/Licenses/libc++ License
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
RAJA_INLINE
void* align(size_t alignment, size_t size, void*& ptr, size_t& space)
{
  void* r = nullptr;
  if (size <= space) {
    char* p1 = static_cast<char*>(ptr);
    char* p2 = reinterpret_cast<char*>(
        reinterpret_cast<size_t>(p1 + (alignment - 1)) & -alignment);
    size_t d = static_cast<size_t>(p2 - p1);
    if (d <= space - size) {
      r = p2;
      ptr = r;
      space -= d;
    }
  }
  return r;
}

}  // end namespace RAJA

#endif
