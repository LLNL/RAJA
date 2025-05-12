/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing implementation for a MPSC 
 *          message queue policy.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_mpsc_queue_HPP
#define RAJA_mpsc_queue_HPP

#include "RAJA/pattern/atomic.hpp"

#include "RAJA/policy/msg_queue/policy.hpp"

namespace RAJA
{
namespace messages
{

template <typename Container>
class queue<Container, RAJA::mpsc_queue>
{
public:
  using policy = RAJA::mpsc_queue;

  using value_type     = typename Container::value_type;
  using size_type      = typename Container::size_type;
  using pointer        = value_type*;
  using const_pointer  = const value_type*;
  using iterator       = pointer;
  using const_iterator = const_pointer;

  queue(Container& container) : m_container{&container}
  {}

  queue(Container* container) : m_container{container}
  {}


  template <typename... Ts>
  RAJA_HOST_DEVICE
  bool try_emplace(Ts&&... args) 
  {
    if (m_container != nullptr) { 
      auto local_size = RAJA::atomicInc<auto_atomic>(&(m_container->m_size));
      if (m_container->m_buf != nullptr && local_size < m_container->m_capacity) {
        m_container[local_size] = value_type(std::forward<Ts>(args)...);
        return true;
      }
    }  

    return false;
  }

private:
  Container* m_container;
};

}
}

#endif  // closing endif for header file include guard
