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

///
/// RAJA Atomic memory order objects
///
constexpr const struct raja_memory_order_relaxed_t
{
  static const std::memory_order value = std::memory_order_relaxed;
} memory_order_relaxed;
///
constexpr const struct raja_memory_order_consume_t
{
  static const std::memory_order value = std::memory_order_consume;
} memory_order_consume;
///
constexpr const struct raja_memory_order_acquire_t
{
  static const std::memory_order value = std::memory_order_acquire;
} memory_order_acquire;
///
constexpr const struct raja_memory_order_release_t
{
  static const std::memory_order value = std::memory_order_release;
} memory_order_release;
///
constexpr const struct raja_memory_order_acq_rel_t
{
  static const std::memory_order value = std::memory_order_acq_rel;
} memory_order_acq_rel;
///
constexpr const struct raja_memory_order_seq_cst_t
{
  static const std::memory_order value = std::memory_order_seq_cst;
} memory_order_seq_cst;

/*!
 ******************************************************************************
 *
 * \brief  Atomic cpu_nonatomic class specialization.
 *
 * Note: Memory_order defaults to relaxed instead of seq_cst.
 *
 ******************************************************************************
 */
template < typename T >
class atomic < T, cpu_nonatomic>
{
public:
  using default_memory_order_t = raja_memory_order_relaxed_t;

  ///
  /// Default constructor default constructs std::atomic<T>.
  ///
  atomic() noexcept = delete;
  //   : m_impl(new T {}),
  //     m_is_copy(false)
  // {

  // }

  ///
  /// Constructor to initialize std::atomic<T> with val.
  ///
  explicit constexpr atomic(T val = T()) noexcept
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
    if (!m_is_copy) delete m_impl;
  }

  ///
  /// Assign val.
  ///
  T operator=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0] = val;
  }
  T operator=(T val) noexcept
  {
    return m_impl[0] = val;
  }

  ///
  /// Store val.
  ///
  template< typename MEM_ORDER >
  void store(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    static_cast<volatile T*>(m_impl)[0] = val;
  }
  template< typename MEM_ORDER >
  void store(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    m_impl[0] = val;
  }

  ///
  /// Load.
  ///
  template< typename MEM_ORDER >
  T load(MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0];
  }
  template< typename MEM_ORDER >
  T load(MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return m_impl[0];
  }

  ///
  /// Load.
  ///
  operator T() const volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0];
  }
  operator T() const noexcept
  {
    return m_impl[0];
  }

  ///
  /// Atomically loads what is stored while replacing it with val.
  /// Returns what was previously stored.
  ///
  template< typename MEM_ORDER >
  T exchange(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    static_cast<volatile T*>(m_impl)[0] = val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T exchange(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    m_impl[0] = val;
    return oldT;
  }

  ///
  /// Atomically compares what is stored with expected and if same replaces what is stored with val other wise loads what is stored into expected.
  /// Returns true if exchange succeeded, false otherwise, expected is overwritten with what was stored.
  /// Note that weak may fail even if the stored value == expected, but may perform better in a loop on some platforms.
  ///
  template< typename MEM_ORDER >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    if (expected == oldT) {
      static_cast<volatile T*>(m_impl)[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    if (expected == oldT) {
      m_impl[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    if (expected == oldT) {
      static_cast<volatile T*>(m_impl)[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    if (expected == oldT) {
      m_impl[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    if (expected == oldT) {
      static_cast<volatile T*>(m_impl)[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) noexcept
  {
    T oldT = m_impl[0];
    if (expected == oldT) {
      m_impl[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    if (expected == oldT) {
      static_cast<volatile T*>(m_impl)[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) noexcept
  {
    T oldT = m_impl[0];
    if (expected == oldT) {
      m_impl[0] = val;
      return true;
    } else {
      expected = oldT;
      return false;
    }
  }

  ///
  /// Atomically operate on the stored value and return the value as it was before this operation.
  ///
  template< typename MEM_ORDER >
  T fetch_add(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    static_cast<volatile T*>(m_impl)[0] += val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_add(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    m_impl[0] += val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_sub(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    static_cast<volatile T*>(m_impl)[0] -= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_sub(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    m_impl[0] -= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_and(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    static_cast<volatile T*>(m_impl)[0] &= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_and(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    m_impl[0] &= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_or(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    static_cast<volatile T*>(m_impl)[0] |= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_or(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    m_impl[0] |= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_xor(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    static_cast<volatile T*>(m_impl)[0] ^= val;
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_xor(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    m_impl[0] ^= val;
    return oldT;
  }

  ///
  /// Atomic min operator, returns the previously stored value
  ///
  template< typename MEM_ORDER >
  T fetch_min(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    if (val < oldT) {
      static_cast<volatile T*>(m_impl)[0] = val;
    }
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_min(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    if (val < oldT) {
      m_impl[0] = val;
    }
    return oldT;
  }

  ///
  /// Atomic max operator, returns the previously stored value
  ///
  template< typename MEM_ORDER >
  T fetch_max(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile T*>(m_impl)[0];
    if (val > oldT) {
      static_cast<volatile T*>(m_impl)[0] = val;
    }
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_max(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl[0];
    if (val > oldT) {
      m_impl[0] = val;
    }
    return oldT;
  }

  ///
  /// Atomic pre-fix operators. Equivalent to fetch_op(1) op 1
  ///
  T operator++() volatile noexcept
  {
    return ++static_cast<volatile T*>(m_impl)[0];
  }
  T operator++() noexcept
  {
    return ++m_impl[0];
  }
  T operator--() volatile noexcept
  {
    return --static_cast<volatile T*>(m_impl)[0];
  }
  T operator--() noexcept
  {
    return --m_impl[0];
  }

  ///
  /// Atomic post-fix operators. Equivalent to fetch_op(1)
  ///
  T operator++(int) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0]++;
  }
  T operator++(int) noexcept
  {
    return m_impl[0]++;
  }
  T operator--(int) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0]--;
  }
  T operator--(int) noexcept
  {
    return m_impl[0]--;
  }

  ///
  /// Atomic operators. Equivalent to fetch_op(val) op val
  ///
  T operator+=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0] += val;
  }
  T operator+=(T val) noexcept
  {
    return m_impl[0] += val;
  }
  T operator-=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0] -= val;
  }
  T operator-=(T val) noexcept
  {
    return m_impl[0] -= val;
  }
  T operator&=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0] &= val;
  }
  T operator&=(T val) noexcept
  {
    return m_impl[0] &= val;
  }
  T operator|=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0] |= val;
  }
  T operator|=(T val) noexcept
  {
    return m_impl[0] |= val;
  }
  T operator^=(T val) volatile noexcept
  {
    return static_cast<volatile T*>(m_impl)[0] ^= val;
  }
  T operator^=(T val) noexcept
  {
    return m_impl[0] ^= val;
  }

private:
  ///
  /// Implementation via pointer to std::atomic.
  ///
  T* m_impl;

  ///
  /// Remember if are copy to free m_impl.
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
  using default_memory_order_t = raja_memory_order_relaxed_t;

  ///
  /// Default constructor default constructs std::atomic<T>.
  ///
  atomic() noexcept = delete;
  //   : m_impl(new std::atomic<T> {}),
  //     m_is_copy(false)
  // {

  // }

  ///
  /// Constructor to initialize std::atomic<T> with val.
  ///
  explicit constexpr atomic(T val = T()) noexcept
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
  /// Atomic destructor frees m_impl if not a copy.
  ///
  ~atomic() noexcept
  {
    if (!m_is_copy) delete m_impl;
  }

  ///
  /// Atomically assign val, equivalent to store(val).
  ///
  T operator=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator=(val);
  }
  T operator=(T val) noexcept
  {
    return m_impl->operator=(val);
  }

  ///
  /// Atomic store val.
  ///
  template< typename MEM_ORDER >
  void store(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    static_cast<volatile std::atomic<T>*>(m_impl)->store(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  void store(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    m_impl->store(val, MEM_ORDER::value);
  }

  ///
  /// Atomic load.
  ///
  template< typename MEM_ORDER >
  T load(MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->load(MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T load(MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return m_impl->load(MEM_ORDER::value);
  }

  ///
  /// Atomically load, equivalent to load().
  ///
  operator T() const volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator T();
  }
  operator T() const noexcept
  {
    return m_impl->operator T();
  }

  ///
  /// Atomically loads what is stored while replacing it with val.
  /// Returns what was previously stored.
  ///
  template< typename MEM_ORDER >
  T exchange(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->exchange(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T exchange(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->exchange(val, MEM_ORDER::value);
  }

  ///
  /// Atomically compares what is stored with expected and if same replaces what is stored with val other wise loads what is stored into expected.
  /// Returns true if exchange succeeded, false otherwise, expected is overwritten with what was stored.
  /// Note that weak may fail even if the stored value == expected, but may perform better in a loop on some platforms.
  ///
  template< typename MEM_ORDER >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->compare_exchange_weak(expected, val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->compare_exchange_weak(expected, val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->compare_exchange_strong(expected, val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->compare_exchange_strong(expected, val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->compare_exchange_weak(expected, val, MEM_ORDER_0::value, MEM_ORDER_1::value);
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) noexcept
  {
    return m_impl->compare_exchange_weak(expected, val, MEM_ORDER_0::value, MEM_ORDER_1::value);
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->compare_exchange_strong(expected, val, MEM_ORDER_0::value, MEM_ORDER_1::value);
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) noexcept
  {
    return m_impl->compare_exchange_strong(expected, val, MEM_ORDER_0::value, MEM_ORDER_1::value);
  }

  ///
  /// Atomically operate on the stored value and return the value as it was before this operation.
  ///
  template< typename MEM_ORDER >
  T fetch_add(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->fetch_add(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_add(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->fetch_add(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_sub(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->fetch_sub(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_sub(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->fetch_sub(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_and(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->fetch_and(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_and(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->fetch_and(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_or(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->fetch_or(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_or(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->fetch_or(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_xor(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->fetch_xor(val, MEM_ORDER::value);
  }
  template< typename MEM_ORDER >
  T fetch_xor(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    return m_impl->fetch_xor(val, MEM_ORDER::value);
  }

  ///
  /// Atomic min operator, returns the previously stored value
  ///
  template< typename MEM_ORDER >
  T fetch_min(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile std::atomic<T>*>(m_impl)->load();
    while (val < oldT && !static_cast<volatile std::atomic<T>*>(m_impl)->compare_exchange_weak(oldT, val, MEM_ORDER::value));
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_min(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl->load();
    while (val < oldT && !m_impl->compare_exchange_weak(oldT, val, MEM_ORDER::value));
    return oldT;
  }

  ///
  /// Atomic max operator, returns the previously stored value
  ///
  template< typename MEM_ORDER >
  T fetch_max(T val, MEM_ORDER m = default_memory_order_t()) volatile noexcept
  {
    T oldT = static_cast<volatile std::atomic<T>*>(m_impl)->load();
    while (val > oldT && !static_cast<volatile std::atomic<T>*>(m_impl)->compare_exchange_weak(oldT, val, MEM_ORDER::value));
    return oldT;
  }
  template< typename MEM_ORDER >
  T fetch_max(T val, MEM_ORDER m = default_memory_order_t()) noexcept
  {
    T oldT = m_impl->load();
    while (val > oldT && !m_impl->compare_exchange_weak(oldT, val, MEM_ORDER::value));
    return oldT;
  }

  ///
  /// Atomic pre-fix operators. Equivalent to fetch_op(1) op 1
  ///
  T operator++() volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator++();
  }
  T operator++() noexcept
  {
    return m_impl->operator++();
  }
  T operator--() volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator--();
  }
  T operator--() noexcept
  {
    return m_impl->operator--();
  }

  ///
  /// Atomic post-fix operators. Equivalent to fetch_op(1)
  ///
  T operator++(int) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator++(0);
  }
  T operator++(int) noexcept
  {
    return m_impl->operator++(0);
  }
  T operator--(int) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator--(0);
  }
  T operator--(int) noexcept
  {
    return m_impl->operator--(0);
  }

  ///
  /// Atomic operators. Equivalent to fetch_op(val) op val
  ///
  T operator+=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator+=(val);
  }
  T operator+=(T val) noexcept
  {
    return m_impl->operator+=(val);
  }
  T operator-=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator-=(val);
  }
  T operator-=(T val) noexcept
  {
    return m_impl->operator-=(val);
  }
  T operator&=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator&=(val);
  }
  T operator&=(T val) noexcept
  {
    return m_impl->operator&=(val);
  }
  T operator|=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator|=(val);
  }
  T operator|=(T val) noexcept
  {
    return m_impl->operator|=(val);
  }
  T operator^=(T val) volatile noexcept
  {
    return static_cast<volatile std::atomic<T>*>(m_impl)->operator^=(val);
  }
  T operator^=(T val) noexcept
  {
    return m_impl->operator^=(val);
  }

private:
  ///
  /// Implementation via pointer to std::atomic.
  ///
  std::atomic<T>* m_impl;

  ///
  /// Remember if are copy to free m_impl.
  ///
  const bool m_is_copy;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
