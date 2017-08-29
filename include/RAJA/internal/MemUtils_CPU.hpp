/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CPU reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CPU_HPP
#define RAJA_MemUtils_CPU_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) \
    || defined(__MINGW32__) || defined(__BORLANDC__)
#define RAJA_PLATFORM_WINDOWS
#include <malloc.h>
#endif

namespace RAJA
{

///
/// Portable aligned memory allocation
///
inline void* allocate_aligned(size_t alignment, size_t size)
{
#if defined(RAJA_HAVE_POSIX_MEMALIGN)
  // posix_memalign available
  void* ret = nullptr;
  int err = posix_memalign(&ret, alignment, size);
  return err ? nullptr : ret;
#elif defined(RAJA_HAVE_ALIGNED_ALLOC)
  return std::aligned_alloc(alignment, size);
#elif defined(RAJA_PLATFORM_WINDOWS)
  return _aligned_malloc(size, alignment);
#else
  char *mem = (char *)malloc(size + alignment + sizeof(void *));
  if (nullptr == mem) return nullptr;
  void **ptr = (void **)((std::uintptr_t)(mem + alignment + sizeof(void *))
                         & ~(alignment - 1));
  // Store the original address one position behind what we give the user.
  ptr[-1] = mem;
  return ptr;
#endif
}


///
/// Portable aligned memory allocation
///
template <typename T>
inline T* allocate_aligned_type(size_t alignment, size_t size)
{
  return reinterpret_cast<T*>(allocate_aligned(alignment, size));
}


///
/// Portable aligned memory free - required for Windows
///
inline void free_aligned(void* ptr)
{
#if defined(RAJA_HAVE_POSIX_MEMALIGN) || defined(RAJA_HAVE_ALIGNED_ALLOC)
  free(ptr);
#elif defined(RAJA_PLATFORM_WINDOWS)
  _aligned_free(ptr);
#else
  // Free the address stored one position behind the user data in ptr.
  // This is valid for pointers allocated with allocate_aligned
  free(((void**)ptr)[-1]);
#endif
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
