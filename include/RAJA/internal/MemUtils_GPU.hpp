/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for GPU reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_GPU_HPP
#define RAJA_MemUtils_GPU_HPP

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

#include <cstddef>
#include <cstdlib>
#include <memory>
//#include <type_traits>
//#include <utility>


#include "RAJA/util/types.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/internal/Span.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#else
#define cudaErrchk(...)
#endif



namespace RAJA
{


template<class T>
class managed_allocator
{
public:
  using value_type = T;

  managed_allocator() {}

  template<class U>
  managed_allocator(const managed_allocator<U>&) {}

  value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal));
#endif

      return result;
    }

  void deallocate(value_type* ptr, size_t)
    {
#if defined(RAJA_ENABLE_CUDA)
      cudaDeviceSynchronize();
      //cudaErrchk(cudaFree(ptr));
#endif

    }
};

template<class T1, class T2>
bool operator==(const managed_allocator<T1>&, const managed_allocator<T2>&)
{
  return true;
}

template<class T1, class T2>
bool operator!=(const managed_allocator<T1>& lhs, const managed_allocator<T2>& rhs)
{
  return !(lhs == rhs);
}



}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
