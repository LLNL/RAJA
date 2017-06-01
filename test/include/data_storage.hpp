/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for storage helpers used in tests.
 *
 ******************************************************************************
 */

#ifndef _STORAGE_HPP_
#define _STORAGE_HPP_

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

#include <cstdio>

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/type_traits.hpp"
#include "RAJA/util/defines.hpp"

namespace internal
{

template <typename ExecPolicy, typename T, bool gpu>
struct storage {
  using type = T;

  static T* alloc(int n)
  {
    return RAJA::allocate_aligned_type<T>(64, n * sizeof(T));
  }

  static void free(T* ptr) { ::free(ptr); }

  static void ready() {}
};

#ifdef RAJA_ENABLE_CUDA

template <typename Exec, typename T>
struct storage<Exec, T, true> {
  using type = T;

  static T* alloc(int n)
  {
    T* ptr;
    ::cudaMallocManaged(&ptr, n * sizeof(T));
    return ptr;
  }

  static void free(T* ptr) { ::cudaFree(ptr); }

  static void ready() { ::cudaDeviceSynchronize(); }
};

#endif
}

struct storage_base {
};

template <typename ExecPolicy, typename T, bool inplace>
struct storage : public storage_base {
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, true> : public storage_base {
  using type = T;

#ifdef RAJA_ENABLE_CUDA
  static constexpr bool UseGPU = RAJA::is_cuda_policy<ExecPolicy>::value;
  using StorageType = typename internal::storage<ExecPolicy, T, UseGPU>;
#else
  using StorageType = typename internal::storage<ExecPolicy, T, false>;
#endif

  storage(int n) : data(StorageType::alloc(n)), elems(n)
  {
    StorageType::ready();
  }

  ~storage() { StorageType::free(data); }
  T* ibegin() { return data; }
  T* iend() { return data + elems; }
  T* obegin() { return data; }
  T* oend() { return data + elems; }
  int size() { return elems; }
  void update() { StorageType::ready(); }

private:
  T* data;
  int elems;
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, false> : public storage_base {
  using type = T;

#ifdef RAJA_ENABLE_CUDA
  using StorageType =
      typename internal::storage<ExecPolicy,
                                 T,
                                 RAJA::is_cuda_policy<ExecPolicy>::value>;
#else
  using StorageType = typename internal::storage<ExecPolicy, T, false>;
#endif

  storage(int n)
      : in(StorageType::alloc(n)), out(StorageType::alloc(n)), elems(n)
  {
    StorageType::ready();
  }
  ~storage()
  {
    StorageType::free(in);
    StorageType::free(out);
  }
  T* ibegin() { return in; }
  T* iend() { return in + elems; }
  T* obegin() { return out; }
  T* oend() { return out + elems; }
  int size() { return elems; }
  void update() { StorageType::ready(); }

private:
  T* in;
  T* out;
  int elems;
};

#endif
