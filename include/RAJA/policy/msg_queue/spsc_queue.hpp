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

#include "RAJA/util/concepts.hpp"
#include "RAJA/pattern/atomic.hpp"

#include "RAJA/policy/msg_queue/policy.hpp"

namespace RAJA
{
namespace messages
{

template<typename Container>
class queue<Container, RAJA::spsc_queue>
{
public:
  using policy = RAJA::spsc_queue;

  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  queue(Container& container) : m_container {&container} {}

  queue(Container* container) : m_container {container} {}

  /// Posts message to queue. This is marked `const` to pass to lambda by
  /// copy. This throws away messages that are over the capacity of the
  /// container.
  template<typename... Ts>
  RAJA_HOST_DEVICE bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      auto local_size = m_container->m_size++;
      if (m_container->m_data != nullptr &&
          local_size < m_container->m_capacity)
      {
        m_container->m_data[local_size] = value_type(std::forward<Ts>(args)...);
        return true;
      }
    }

    return false;
  }

private:
  Container* m_container;
};

template<typename Container>
class queue<Container, RAJA::spsc_queue_overwrite>
{
public:
  using policy = RAJA::spsc_queue_overwrite;

  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  queue(Container& container) : m_container {&container} {}

  queue(Container* container) : m_container {container} {}

  /// Posts message to queue. This is marked `const` to pass to lambda by
  /// copy. This overwrites previously stored messages once the number of
  /// messages are over the capacity of the container.
  template<typename... Ts>
  RAJA_HOST_DEVICE bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      auto local_size = m_container->m_size++;
      if (m_container->m_data != nullptr)
      {
        m_container->m_data[local_size % m_container->m_capacity] = value_type(std::forward<Ts>(args)...);
        return true;
      }
    }

    return false;
  }

private:
  Container* m_container;
};

}  // namespace messages
}  // namespace RAJA

#endif  // closing endif for header file include guard
