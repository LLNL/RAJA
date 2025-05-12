/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a GPU to CPU message handler class.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MESSAGES_HPP
#define RAJA_MESSAGES_HPP

#include <algorithm>
#include <functional>
#include "RAJA/policy/msg_queue.hpp"

// TODO: should these use the RAJA headers instead?
#include "camp/tuple.hpp"
#include "camp/resource.hpp"

namespace RAJA
{
namespace detail 
{	
  ///
  /// Queue for storing messages. Fills buffer up to capacity.
  /// Once at capacity, messages are discarded.
  ///
  template <typename T>
  class queue 
  {
  public:
    using value_type     = T;
    using size_type      = unsigned long long;
    using pointer        = value_type*;
    using const_pointer  = const value_type*;
    using iterator       = pointer;
    using const_iterator = const_pointer;

    queue() : m_capacity{0}, m_size{0}, m_buf{nullptr} 
    {}
    queue(size_type capacity, pointer buf) : 
      m_capacity{capacity}, m_size{0}, m_buf{buf} 
    {}

    constexpr pointer data() noexcept {
      return m_buf;
    }

    constexpr const_pointer data() const noexcept {
      return m_buf;
    }

    constexpr size_type capacity() const noexcept {
      return m_capacity;
    }

    constexpr size_type size() const noexcept {
      return std::min(m_capacity, m_size);
    }

    constexpr bool empty() const noexcept {
      return size() == 0;
    }

    constexpr iterator begin() noexcept { 
      return data(); 
    }

    constexpr const_iterator begin() const noexcept { 
      return const_iterator(data()); 
    }

    constexpr const_iterator cbegin() const noexcept { 
      return const_iterator(data()); 
    }

    constexpr iterator end() noexcept {
      return data()+size(); 
    }

    constexpr const_iterator end() const noexcept { 
      return const_iterator(data()+size()); 
    }

    constexpr const_iterator cend() const noexcept   { 
      return const_iterator(data()+size()); 
    }

    void clear() noexcept
    {
      m_size = 0;
    }

    size_type m_capacity;
    size_type m_size;
    pointer m_buf;
  };
} // end of detail namespace 

  template <typename Callable>
  class message_handler;

  ///
  /// Provides a way to handle messages from a GPU. This currently
  /// stores messages from the GPU and then calls a callback 
  /// function from the host.
  ///
  /// Note: 
  /// Currently, this forces a synchronize prior to calling 
  /// the callback function or testing if there are any messages.
  ///
  template <typename R, typename... Args>
  class message_handler<R(Args...)>
  {
  public:
    using message       = camp::tuple<std::decay_t<Args>...>;  
    using msg_queue     = detail::queue<message>;
    using callback_type = std::function<R(Args...)>;

  public:
    template <typename Callable>
    message_handler(const std::size_t num_messages, Callable c) 
      : m_res{camp::resources::Host()}, 
        m_queue{num_messages, m_res.allocate<message>(num_messages,
            camp::resources::MemoryAccess::Pinned)}, 
        m_callback{c}
    {}  

    template <typename Resource, typename Callable>
    message_handler(const std::size_t num_messages, Resource res, 
                    Callable c) 
      : m_res{res}, 
        m_queue{num_messages, m_res.allocate<message>(num_messages,
            camp::resources::MemoryAccess::Pinned)}, 
        m_callback{c}
    {}  

    ~message_handler() 
    {
      m_res.wait();
      m_res.deallocate(m_queue.data(), camp::resources::MemoryAccess::Pinned); 
    }

    // Doesn't support copying 
    message_handler(const message_handler&) = delete;
    message_handler& operator=(const message_handler&) = delete;

    // TODO need proper move support 
    // Move ctor/operator
    message_handler(message_handler&&) = delete;
    message_handler& operator=(message_handler&&) = delete;

    template <typename Policy>
    RAJA::messages::queue<msg_queue, Policy> get_queue()
    {
      return RAJA::messages::queue<msg_queue, Policy>{m_queue};
    } 

    void clear()
    {
      m_res.wait();   
      m_queue.clear();
    }

    bool test_any()
    {
      m_res.wait();   
      return !m_queue.empty(); 
    }

    void wait_all()
    {
      if (test_any()) {
        for (const auto& msg: m_queue) {
          camp::apply(m_callback, msg);     
        }
        clear();
      }
    }

  private:
    camp::resources::Resource m_res;
    msg_queue m_queue;
    callback_type m_callback;
  }; 
}

#endif /* RAJA_MESSAGES_HPP */
