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

#include <atomic>

namespace RAJA
{

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
struct cpu_atomic {
};

///
/// Not actually atomic CPU policy, to avoid use of atomics in sequential
/// loops where it is not necessary.
///
struct cpu_nonatomic {
};

///
/// Atomic class declaration
///
template < typename T, typename POLICY >
class atomic;

/*!
 ******************************************************************************
 *
 * \brief  Atomic cpu_nonatomic class specialization.
 *
 * Note: For now memory_order is always memory_order_relaxed.
 *
 ******************************************************************************
 */
template < typename T >
class atomic < T, cpu_nonatomic>
{
public:
  static const std::memory_order default_memory_order = std::memory_order_relaxed;

  ///
  /// Default constructor default constructs std::atomic<T>.
  ///
  atomic() noexcept
    : m_impl(new T {}),
      m_is_copy(false)
  {

  }

  ///
  /// Constructor to initialize std::atomic<T> with val.
  ///
  constexpr atomic(T val) noexcept
    : m_impl(new T {val}),
      m_is_copy(false)
  {

  }

  ///
  /// Atomic copy constructor.
  ///
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
  /// Atomic destructor frees m_impl if not a copy.
  ///
  ~atomic() noexcept
  {
    if (!m_is_copy) delete this->m_impl;
  }

  ///
  /// Assign val.
  ///
  T operator=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0] = val;
  }
  T operator=(T val) noexcept
  {
    return this->m_impl[0] = val;
  }

  ///
  /// Store val.
  ///
  void store(T val) volatile noexcept
  {
    static_cast<volatile T*>(this->m_impl)[0] = val;
  }
  void store(T val) noexcept
  {
    this->m_impl[0] = val;
  }

  ///
  /// Load.
  ///
  T load() const volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0];
  }
  T load() const noexcept
  {
    return this->m_impl[0];
  }

  ///
  /// Load.
  ///
  operator T() const volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0];
  }
  operator T() const noexcept
  {
    return this->m_impl[0];
  }

  ///
  /// Atomically loads what is stored while replacing it with val.
  /// Returns what was previously stored.
  ///
  T exchange(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    static_cast<volatile T*>(this->m_impl)[0] = val;
    return oldT;
  }
  T exchange(T val) noexcept
  {
    T oldT = this->m_impl[0];
    this->m_impl[0] = val;
    return oldT;
  }

  ///
  /// Atomically compares what is stored with expected and if same replaces what is stored with val other wise loads what is stored into expected.
  /// Returns true if exchange succeeded, false otherwise, expected is overwritten with what was stored.
  /// Note that weak may fail even if the stored value == expected, but may perform better in a loop on some platforms.
  ///
  bool compare_exchange_weak(T& expected, T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    if (expected == oldT) {
      static_cast<volatile T*>(this->m_impl)[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  bool compare_exchange_weak(T& expected, T val) noexcept
  {
    T oldT = this->m_impl[0];
    if (expected == oldT) {
      this->m_impl[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  bool compare_exchange_strong(T& expected, T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    if (expected == oldT) {
      static_cast<volatile T*>(this->m_impl)[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  bool compare_exchange_strong(T& expected, T val) noexcept
  {
    T oldT = this->m_impl[0];
    if (expected == oldT) {
      this->m_impl[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }

  ///
  /// Atomically operate on the stored value and return the value as it was before this operation.
  ///
  T fetch_add(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    static_cast<volatile T*>(this->m_impl)[0] += val;
    return oldT;
  }
  T fetch_add(T val) noexcept
  {
    T oldT = this->m_impl[0];
    this->m_impl[0] += val;
    return oldT;
  }
  T fetch_sub(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    static_cast<volatile T*>(this->m_impl)[0] -= val;
    return oldT;
  }
  T fetch_sub(T val) noexcept
  {
    T oldT = this->m_impl[0];
    this->m_impl[0] -= val;
    return oldT;
  }
  T fetch_and(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    static_cast<volatile T*>(this->m_impl)[0] &= val;
    return oldT;
  }
  T fetch_and(T val) noexcept
  {
    T oldT = this->m_impl[0];
    this->m_impl[0] &= val;
    return oldT;
  }
  T fetch_or(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    static_cast<volatile T*>(this->m_impl)[0] |= val;
    return oldT;
  }
  T fetch_or(T val) noexcept
  {
    T oldT = this->m_impl[0];
    this->m_impl[0] |= val;
    return oldT;
  }
  T fetch_xor(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    static_cast<volatile T*>(this->m_impl)[0] ^= val;
    return oldT;
  }
  T fetch_xor(T val) noexcept
  {
    T oldT = this->m_impl[0];
    this->m_impl[0] ^= val;
    return oldT;
  }

  ///
  /// Atomic pre-fix operators. Equivalent to fetch_op(1) op 1
  ///
  T operator++() volatile noexcept
  {
    return ++static_cast<volatile T*>(this->m_impl)[0];
  }
  T operator++() noexcept
  {
    return ++this->m_impl[0];
  }
  T operator--() volatile noexcept
  {
    return --static_cast<volatile T*>(this->m_impl)[0];
  }
  T operator--() noexcept
  {
    return --this->m_impl[0];
  }

  ///
  /// Atomic post-fix operators. Equivalent to fetch_op(1)
  ///
  T operator++(int) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0]++;
  }
  T operator++(int) noexcept
  {
    return this->m_impl[0]++;
  }
  T operator--(int) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0]--;
  }
  T operator--(int) noexcept
  {
    return this->m_impl[0]--;
  }

  ///
  /// Atomic operators. Equivalent to fetch_op(val) op val
  ///
  T operator+=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0] += val;
  }
  T operator+=(T val) noexcept
  {
    return this->m_impl[0] += val;
  }
  T operator-=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0] -= val;
  }
  T operator-=(T val) noexcept
  {
    return this->m_impl[0] -= val;
  }
  T operator&=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0] &= val;
  }
  T operator&=(T val) noexcept
  {
    return this->m_impl[0] &= val;
  }
  T operator|=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0] |= val;
  }
  T operator|=(T val) noexcept
  {
    return this->m_impl[0] |= val;
  }
  T operator^=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(this->m_impl)[0] ^= val;
  }
  T operator^=(T val) noexcept
  {
    return this->m_impl[0] ^= val;
  }

  ///
  /// Atomic min operator, returns the previously stored value
  ///
  T fetch_min(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    if (val < oldT) {
      static_cast<volatile T*>(this->m_impl)[0] = val;
    }
    return oldT;
  }
  T fetch_min(T val) noexcept
  {
    T oldT = this->m_impl[0];
    if (val < oldT) {
      this->m_impl[0] = val;
    }
    return oldT;
  }

  ///
  /// Atomic max operator, returns the previously stored value
  ///
  T fetch_max(T val) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(this->m_impl)[0];
    if (val > oldT) {
      static_cast<volatile T*>(this->m_impl)[0] = val;
    }
    return oldT;
  }
  T fetch_max(T val) noexcept
  {
    T oldT = this->m_impl[0];
    if (val > oldT) {
      this->m_impl[0] = val;
    }
    return oldT;
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
template < typename T >
class atomic < T, cpu_atomic>
{
public:
  static const std::memory_order default_memory_order = std::memory_order_relaxed;

  ///
  /// Default constructor default constructs std::atomic<T>.
  ///
  atomic() noexcept
    : m_impl(new std::atomic<T> {}),
      m_is_copy(false)
  {

  }

  ///
  /// Constructor to initialize std::atomic<T> with val.
  ///
  constexpr atomic(T val) noexcept
    : m_impl(new std::atomic<T> {val}),
      m_is_copy(false)
  {

  }

  ///
  /// Atomic copy constructor.
  ///
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
  ~atomic() noexcept
  {
    if (!m_is_copy) delete this->m_impl;
  }

  ///
  /// Atomically assign val, equivalent to store(val).
  ///
  T operator=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator=(val);
  }
  T operator=(T val) noexcept
  {
    return this->m_impl->operator=(val);
  }

  ///
  /// Atomic store val.
  ///
  void store(T val) volatile noexcept
  {
    static_cast<volatile std::atomic<T>*>(this->m_impl)->store(val, default_memory_order);
  }
  void store(T val) noexcept
  {
    this->m_impl->store(val, default_memory_order);
  }

  ///
  /// Atomic load.
  ///
  T load() const volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->load(default_memory_order);
  }
  T load() const noexcept
  {
    return this->m_impl->load(default_memory_order);
  }

  ///
  /// Atomically load, equivalent to load().
  ///
  operator T() const volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator T();
  }
  operator T() const noexcept
  {
    return this->m_impl->operator T();
  }

  ///
  /// Atomically loads what is stored while replacing it with val.
  /// Returns what was previously stored.
  ///
  T exchange(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->exchange(val, default_memory_order);
  }
  T exchange(T val) noexcept
  {
    return this->m_impl->exchange(val, default_memory_order);
  }

  ///
  /// Atomically compares what is stored with expected and if same replaces what is stored with val other wise loads what is stored into expected.
  /// Returns true if exchange succeeded, false otherwise, expected is overwritten with what was stored.
  /// Note that weak may fail even if the stored value == expected, but may perform better in a loop on some platforms.
  ///
  bool compare_exchange_weak(T& expected, T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->compare_exchange_weak(expected, val, default_memory_order);
  }
  bool compare_exchange_weak(T& expected, T val) noexcept
  {
    return this->m_impl->compare_exchange_weak(expected, val, default_memory_order);
  }
  bool compare_exchange_strong(T& expected, T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->compare_exchange_strong(expected, val, default_memory_order);
  }
  bool compare_exchange_strong(T& expected, T val) noexcept
  {
    return this->m_impl->compare_exchange_strong(expected, val, default_memory_order);
  }

  ///
  /// Atomically operate on the stored value and return the value as it was before this operation.
  ///
  T fetch_add(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->fetch_add(val, default_memory_order);
  }
  T fetch_add(T val) noexcept
  {
    return this->m_impl->fetch_add(val, default_memory_order);
  }
  T fetch_sub(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->fetch_sub(val, default_memory_order);
  }
  T fetch_sub(T val) noexcept
  {
    return this->m_impl->fetch_sub(val, default_memory_order);
  }
  T fetch_and(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->fetch_and(val, default_memory_order);
  }
  T fetch_and(T val) noexcept
  {
    return this->m_impl->fetch_and(val, default_memory_order);
  }
  T fetch_or(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->fetch_or(val, default_memory_order);
  }
  T fetch_or(T val) noexcept
  {
    return this->m_impl->fetch_or(val, default_memory_order);
  }
  T fetch_xor(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->fetch_xor(val, default_memory_order);
  }
  T fetch_xor(T val) noexcept
  {
    return this->m_impl->fetch_xor(val, default_memory_order);
  }

  ///
  /// Atomic pre-fix operators. Equivalent to fetch_op(1) op 1
  ///
  T operator++() volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator++();
  }
  T operator++() noexcept
  {
    return this->m_impl->operator++();
  }
  T operator--() volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator--();
  }
  T operator--() noexcept
  {
    return this->m_impl->operator--();
  }

  ///
  /// Atomic post-fix operators. Equivalent to fetch_op(1)
  ///
  T operator++(int) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator++(0);
  }
  T operator++(int) noexcept
  {
    return this->m_impl->operator++(0);
  }
  T operator--(int) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator--(0);
  }
  T operator--(int) noexcept
  {
    return this->m_impl->operator--(0);
  }

  ///
  /// Atomic operators. Equivalent to fetch_op(val) op val
  ///
  T operator+=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator+=(val);
  }
  T operator+=(T val) noexcept
  {
    return this->m_impl->operator+=(val);
  }
  T operator-=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator-=(val);
  }
  T operator-=(T val) noexcept
  {
    return this->m_impl->operator-=(val);
  }
  T operator&=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator&=(val);
  }
  T operator&=(T val) noexcept
  {
    return this->m_impl->operator&=(val);
  }
  T operator|=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator|=(val);
  }
  T operator|=(T val) noexcept
  {
    return this->m_impl->operator|=(val);
  }
  T operator^=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(this->m_impl)->operator^=(val);
  }
  T operator^=(T val) noexcept
  {
    return this->m_impl->operator^=(val);
  }

  ///
  /// Atomic min operator, returns the previously stored value
  ///
  T fetch_min(T val) volatile noexcept
  {
    T oldT = static_cast<volatile std::atomic<T>*>(this->m_impl)->load();
    while (val < oldT && !static_cast<volatile std::atomic<T>*>(this->m_impl)->compare_exchange_weak(oldT, val, default_memory_order));
    return oldT;
  }
  T fetch_min(T val) noexcept
  {
    T oldT = this->m_impl->load();
    while (val < oldT && !this->m_impl->compare_exchange_weak(oldT, val, default_memory_order));
    return oldT;
  }

  ///
  /// Atomic max operator, returns the previously stored value
  ///
  T fetch_max(T val) volatile noexcept
  {
    T oldT = static_cast<volatile std::atomic<T>*>(this->m_impl)->load();
    while (val > oldT && !static_cast<volatile std::atomic<T>*>(this->m_impl)->compare_exchange_weak(oldT, val, default_memory_order));
    return oldT;
  }
  T fetch_max(T val) noexcept
  {
    T oldT = this->m_impl->load();
    while (val > oldT && !this->m_impl->compare_exchange_weak(oldT, val, default_memory_order));
    return oldT;
  }

private:
  ///
  /// Implementation via pointer to std::atomic.
  ///
  std::atomic<T>* m_impl;

  ///
  /// Remember if are copy to free this->m_impl.
  ///
  const bool m_is_copy;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
