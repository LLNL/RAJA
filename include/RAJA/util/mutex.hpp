/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing functionality similar to std mutex header.
*
******************************************************************************
*/

#ifndef RAJA_util_mutex_HPP
#define RAJA_util_mutex_HPP

#include "RAJA/config.hpp"

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

#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
namespace omp
{

//! class wrapping omp_lock_t with std::mutex interface
class mutex {
public:
  using native_handle_type = omp_lock_t;

  mutex()
  {
    omp_init_lock(&m_lock);
  }

  mutex( const mutex& ) = delete;
  mutex( mutex&& ) = delete;
  mutex& operator=( const mutex& ) = delete;
  mutex& operator=( mutex&& ) = delete;

  void lock()
  {
    omp_set_lock(&m_lock);
  }

  bool try_lock()
  {
    return omp_test_lock(&m_lock) != 0;
  }

  void unlock()
  {
    omp_unset_lock(&m_lock);
  }

  native_handle_type& native_handle()
  {
    return m_lock;
  }

  ~mutex()
  {
    omp_destroy_lock(&m_lock);
  }

private:
  native_handle_type m_lock;
};

} // namespace omp
#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)

//! class providing functionality of std::lock_guard
template < typename mutex_type >
class lock_guard {
public:
  
  explicit lock_guard( mutex_type& m )
    : m_mutex(m)
  {
    m_mutex.lock();
  }

  lock_guard( const lock_guard& ) = delete;
  lock_guard( lock_guard&& ) = delete;
  lock_guard& operator=( const lock_guard& ) = delete;
  lock_guard& operator=( lock_guard&& ) = delete;

  ~lock_guard()
  {
    m_mutex.unlock();
  }

private:
  mutex_type& m_mutex;
};

}  // namespace RAJA

#endif  // closing endif for header file include guard
