/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple vector template class that enables
 *          RAJA to be used with or without the C++ STL.
 *
 ******************************************************************************
 */

#ifndef RAJA_CudaManagedAllocator_HXX
#define RAJA_CudaManagedAllocator_HXX

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

#include <algorithm>

#include "RAJA/config.hxx"
#include "RAJA/internal/exec-cuda/raja_cudaerrchk.hxx"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class template that provides a simple allocator
 *         that allocated in unified memory when cuda is enabled
 *         and uses regular allocation otherwise
 *
 ******************************************************************************
 */
template <class T>
class cuda_managed_allocator {
public:
  using value_type = T;
  //cuda_managed_allocator() noexcept;
  //template <class U> cuda_managed_allocator (const cuda_managed_allocator<U>&) noexcept;

  T* allocate (std::size_t n) {
#if defined(RAJA_ENABLE_CUDA)
    T* result = nullptr;
    cudaErrchk(cudaMallocManaged((void**)&result, n * sizeof(T), cudaMemAttachGlobal));
    cudaErrchk(cudaMemset(result, 0, n * sizeof(T)));
    cudaErrchk(cudaDeviceSynchronize());
    return result;
#else
    return new T[n];
#endif
  }


  void deallocate (T* p, std::size_t n) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(p));
    cudaErrchk(cudaDeviceSynchronize());
#else
    delete[] p;
#endif
}


}; //end class cuda_managed_allocator;


//template <class T, class U>
//constexpr bool operator== (const cuda_managed_allocator<T>&, const cuda_managed_allocator<U>&) noexcept;

//template <class T, class U>
//constexpr bool operator!= (const cuda_managed_allocator<T>&, const cuda_managed_allocator<U>&) noexcept;


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
