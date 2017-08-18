/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_util_atomic_HPP
#define RAJA_util_atomic_HPP

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
#include "RAJA/util/defines.hpp"
#include "RAJA/util/atomic/builtin.hpp"
#include "RAJA/util/atomic/auto.hpp"


namespace RAJA
{



/*
 * Atomic operation functions
 *
 * The dispatch of all of these is:
 *
 * T atomicAdd<Policy>(T *acc, T value)      -- User facing API
 *
 * calls
 *
 * T atomicAdd(Policy{}, T *acc, T value)    -- Policy specific implementation
 *
 *
 * With the exception of the auto_atomic policy which then calls the
 * "appropriate" policy implementation.
 *
 *
 * Current supported policies include:
 *   cuda_atomic       -- Only atomic supported in CUDA device functions
 *
 *   omp_atomic        -- Available (and default) when OpenMP is enabled
 *                        these are safe inside and outside of OMP parallel
 *                        regions
 *
 *   builtin_atomic    -- Use the (nonstandard) __sync_fetch_and_XXX functions
 *
 *   seq_atomic        -- Non-atomic, does an unprotected (raw) operation
 *
 *
 * The implementation code lives in:
 * RAJA/util/atomic/auto.hpp    -- for auto_atomic
 * RAJA/util/atomic/builtin.hpp -- for builtin_atomic
 * RAJA/policy/XXX/atomic.hpp   -- for omp_atomic, cuda_atomic, etc.
 *
 */



template<typename Policy, typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
T atomicAdd(T *acc, T value){
  return RAJA::atomicAdd(Policy{}, acc, value);
}


template<typename Policy, typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
T atomicSub(T *acc, T value){
  return RAJA::atomicSub(Policy{}, acc, value);
}





/*!
 * Atomic operation object
 *
 * @TODO DOCUMENT THIS
 */
template<typename T, typename Policy = RAJA::auto_atomic>
class AtomicRef {
  public:
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    explicit
    AtomicRef(T *value_ptr) : m_value_ptr(value_ptr){};


    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    AtomicRef(AtomicRef<T, Policy> const &c) : m_value_ptr(c.m_value_ptr){};


    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    AtomicRef(AtomicRef<T, Policy> &c) : m_value_ptr(c.m_value_ptr){};


    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    T *getPointer() const {
      return m_value_ptr;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    AtomicRef const &operator++()const{
      RAJA::atomicAdd<Policy>(m_value_ptr, (T)1);
      return *this;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    AtomicRef const &operator++(int)const{
      RAJA::atomicAdd<Policy>(m_value_ptr, (T)1);
      return *this;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    AtomicRef const &operator--()const{
      RAJA::atomicSub<Policy>(m_value_ptr, (T)1);
      return *this;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    AtomicRef const &operator--(int)const{
      RAJA::atomicSub<Policy>(m_value_ptr, (T)1);
      return *this;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    AtomicRef const &operator+=(T rhs)const{
      RAJA::atomicAdd<Policy>(m_value_ptr, rhs);
      return *this;
    }


    RAJA_INLINE
    RAJA_HOST_DEVICE
    AtomicRef const &operator-=(T rhs)const{
      RAJA::atomicSub<Policy>(m_value_ptr, rhs);
      return *this;
    }


  private:
    T *m_value_ptr;
};


}  // namespace RAJA

#endif
