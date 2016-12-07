/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic class.
 *
 ******************************************************************************
 */

#ifndef RAJA_Atomic_HXX
#define RAJA_Atomic_HXX

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"
#include "RAJA/Atomic.hxx"

#include <type_traits>

namespace RAJA
{

///
/// Max number of cuda tomic variables.
///
#define RAJA_MAX_ATOMIC_CUDA_VARS 8

//
//////////////////////////////////////////////////////////////////////
//
// Atomic policy
//
//////////////////////////////////////////////////////////////////////
//

///
/// Basic cpu use only atomic policy.
///
template < bool Async >
struct cuda_atomic {
};

/*!
 ******************************************************************************
 *
 * \brief  Atomic cpu_atomic class specialization.
 *
 * Note: For now memory_order is always memory_order_relaxed so you should
 *       make any assumptions about the order of memory update visibility in
 *       different threads, a.k.a. these are meant mostly for atomic counters,
 *       not for any form of thread synchronization.
 *
 ******************************************************************************
 */
template < typename T, bool Async >
class atomic < T, cuda_atomic<Async> >
{
public:
  static const std::memory_order default_memory_order = std::memory_order_relaxed;

  ///
  /// Default constructor default constructs T.
  ///
  __host__
  atomic() noexcept
    : m_impl(getCudaAtomicptr<T>(T {})),
      m_is_copy(false)
  {

  }

  ///
  /// Constructor to initialize T with val.
  ///
  __host__
  constexpr atomic(T val) noexcept
    : m_impl(getCudaAtomicptr<T>( T {val})),
      m_is_copy(false)
  {

  }

  ///
  /// Atomic copy constructor.
  ///
  __host__ __device__
  atomic(const atomic& other) noexcept
    : m_impl(other.m_impl),
      m_is_copy(true)
  {

  }

  ///
  /// Atomic copy assignment operator deleted.
  ///
  atomic& operator=(const atomic&) = delete;
  atomic& operator=(const atomic&) volatile = delete;

  ///
  /// Atomic destructor frees this->m_impl if not a copy.
  ///
  __host__ __device__
  ~atomic() noexcept
  {
#if !defined(__CUDA_ARCH__)
    if (!m_is_copy) releaseCudaAtomicptr(this->m_impl);
#endif
  }

  ///
  /// Atomically assign val, equivalent to store(val).
  ///
  __host__ __device__
  T operator=(T val) volatile noexcept
  {
    _atomicExch(static_cast<volatile T*>(this->m_impl), val);
    return val;
  }
  __host__ __device__
  T operator=(T val) noexcept
  {
    _atomicExch(this->m_impl, val);
    return val;
  }

  ///
  /// Atomic store val.
  ///
  __host__ __device__
  void store(T val) volatile noexcept
  {
    _atomicExch(static_cast<volatile T*>(this->m_impl), val);
  }
  __host__ __device__
  void store(T val) noexcept
  {
    _atomicExch(this->m_impl, val);
  }

  ///
  /// Atomic load.
  ///
  __host__ __device__
  T load() const volatile noexcept
  {
    return _atomicCAS(static_cast<volatile T*>(this->m_impl), T{});
  }
  __host__ __device__
  T load() const noexcept
  {
    return _atomicCAS(this->m_impl, T{});
  }

  ///
  /// Atomically load, equivalent to load().
  ///
  __host__ __device__
  operator T() const volatile noexcept
  {
    return _atomicCAS(static_cast<volatile T*>(this->m_impl), T{});
  }
  __host__ __device__
  operator T() const noexcept
  {
    return _atomicCAS(this->m_impl, T{});
  }

  ///
  /// Atomically loads what is stored while replacing it with val.
  /// Returns what was previously stored.
  ///
  __device__
  T exchange(T val) volatile noexcept
  {
    return _atomicExch(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T exchange(T val) noexcept
  {
    return _atomicExch(this->m_impl, val);
  }

  ///
  /// Atomically compares what is stored with expected and if same replaces what is stored with val other wise loads what is stored into expected.
  /// Returns true if exchange succeeded, false otherwise, expected is overwritten with what was stored.
  /// Note that weak may fail even if the stored value == expected, but may perform better in a loop on some platforms.
  ///
  __device__
  bool compare_exchange_weak(T& expected, T val) volatile noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(static_cast<volatile T*>(this->m_impl), expected, val);
    return old_exp == expected;
  }
  __device__
  bool compare_exchange_weak(T& expected, T val) noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(this->m_impl, expected, val);
    return old_exp == expected;
  }
  __device__
  bool compare_exchange_strong(T& expected, T val) volatile noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(static_cast<volatile T*>(this->m_impl), expected, val);
    return old_exp == expected;
  }
  __device__
  bool compare_exchange_strong(T& expected, T val) noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(this->m_impl, expected, val);
    return old_exp == expected;
  }

  ///
  /// Atomically operate on the stored value and return the value as it was before this operation.
  ///
  __device__
  T fetch_add(T val) volatile noexcept
  {
    return _atomicAdd(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_add(T val) noexcept
  {
    return _atomicAdd(this->m_impl, val);
  }
  __device__
  T fetch_sub(T val) volatile noexcept
  {
    return _atomicSub(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_sub(T val) noexcept
  {
    return _atomicSub(this->m_impl, val);
  }
  __device__
  T fetch_and(T val) volatile noexcept
  {
    return _atomicAnd(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_and(T val) noexcept
  {
    return _atomicAnd(this->m_impl, val);
  }
  __device__
  T fetch_or(T val) volatile noexcept
  {
    return _atomicOr(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_or(T val) noexcept
  {
    return _atomicOr(this->m_impl, val);
  }
  __device__
  T fetch_xor(T val) volatile noexcept
  {
    return _atomicXor(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_xor(T val) noexcept
  {
    return _atomicXor(this->m_impl, val);
  }

  ///
  /// Atomic pre-fix operators. Equivalent to fetch_op(1) op 1
  ///
  __device__
  T operator++() volatile noexcept
  {
    return _atomicAdd(static_cast<volatile T*>(this->m_impl), static_cast<T>(1)) + static_cast<T>(1);
  }
  __device__
  T operator++() noexcept
  {
    return _atomicAdd(this->m_impl, static_cast<T>(1)) + static_cast<T>(1);
  }
  __device__
  T operator--() volatile noexcept
  {
    return _atomicSub(static_cast<volatile T*>(this->m_impl), static_cast<T>(1)) - static_cast<T>(1);
  }
  __device__
  T operator--() noexcept
  {
    return _atomicSub(this->m_impl, static_cast<T>(1)) - static_cast<T>(1);
  }

  ///
  /// Atomic post-fix operators. Equivalent to fetch_op(1)
  ///
  __device__
  T operator++(int) volatile noexcept
  {
    return _atomicAdd(static_cast<volatile T*>(this->m_impl), static_cast<T>(1));
  }
  __device__
  T operator++(int) noexcept
  {
    return _atomicAdd(this->m_impl, static_cast<T>(1));
  }
  __device__
  T operator--(int) volatile noexcept
  {
    return _atomicSub(static_cast<volatile T*>(this->m_impl), static_cast<T>(1));
  }
  __device__
  T operator--(int) noexcept
  {
    return _atomicSub(this->m_impl, static_cast<T>(1));
  }

  ///
  /// Atomic operators. Equivalent to fetch_op(val) op val
  ///
  __device__
  T operator+=(T val) volatile noexcept
  {
    return _atomicAdd(static_cast<volatile T*>(this->m_impl), val) + val;
  }
  __device__
  T operator+=(T val) noexcept
  {
    return _atomicAdd(this->m_impl, val) + val;
  }
  __device__
  T operator-=(T val) volatile noexcept
  {
    return _atomicSub(static_cast<volatile T*>(this->m_impl), val) - val;
  }
  __device__
  T operator-=(T val) noexcept
  {
    return _atomicSub(this->m_impl, val) - val;
  }
  __device__
  T operator&=(T val) volatile noexcept
  {
    return _atomicAnd(static_cast<volatile T*>(this->m_impl), val) & val;
  }
  __device__
  T operator&=(T val) noexcept
  {
    return _atomicAnd(this->m_impl, val) & val;
  }
  __device__
  T operator|=(T val) volatile noexcept
  {
    return _atomicOr(static_cast<volatile T*>(this->m_impl), val) | val;
  }
  __device__
  T operator|=(T val) noexcept
  {
    return _atomicOr(this->m_impl, val) | val;
  }
  __device__
  T operator^=(T val) volatile noexcept
  {
    return _atomicXor(static_cast<volatile T*>(this->m_impl), val) ^ val;
  }
  __device__
  T operator^=(T val) noexcept
  {
    return _atomicXor(this->m_impl, val) ^ val;
  }

  ///
  /// Atomic min operator, returns the previously stored value
  ///
  __device__
  T fetch_min(T val) volatile noexcept
  {
    return _atomicMin(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_min(T val) noexcept
  {
    return _atomicMin(this->m_impl, val);
  }

  ///
  /// Atomic max operator, returns the previously stored value
  ///
  __device__
  T fetch_max(T val) volatile noexcept
  {
    return _atomicMax(static_cast<volatile T*>(this->m_impl), val);
  }
  __device__
  T fetch_max(T val) noexcept
  {
    return _atomicMax(this->m_impl, val);
  }

private:
  ///
  /// Implementation via pointer to std::atomic.
  ///
  T* m_impl;

  ///
  /// Remember if are copy to free this->m_impl.
  ///
  const bool m_is_copy;

#if !defined(__CUDA_ARCH__)
  static_assert(std::is_arithmetic<T>::value,
                "Error: type must be arithmetic");
  static_assert(sizeof(T) <= RAJA_CUDA_ATOMIC_VAR_MAXSIZE,
                "Error: type must be of size <= "
                RAJA_STRINGIFY_MACRO(RAJA_CUDA_ATOMIC_VAR_MAXSIZE));
#endif
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
