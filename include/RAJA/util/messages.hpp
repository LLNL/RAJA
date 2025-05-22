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
#include <memory>

#include "RAJA/policy/msg_queue.hpp"

// TODO: should these use the RAJA headers instead?
#include "camp/tuple.hpp"
#include "camp/resource.hpp"

namespace RAJA
{
  ///
  /// Owning wrapper for a message queue. This is used for ownership
  /// of the message queue and is a move-only class. For getting a view-like
  /// class, use the `get_queue` member function, which allows copying.
  ///
  template <typename T>
  class message_bus
  {
  public:
    using value_type     = T;
    using size_type      = unsigned long long;
    using pointer        = value_type*;
    using const_pointer  = const value_type*;
    using iterator       = value_type*;
    using const_iterator = const value_type*;
    using resource_type  = camp::resources::Resource; 

  private:
    // Internal classes
    struct queue
    {
      using value_type     = T;
      using size_type      = unsigned long long;
      using pointer        = value_type*;
      using const_pointer  = const value_type*;
      using iterator       = value_type*;
      using const_iterator = const value_type*;

      size_type m_size{0};
      size_type m_capacity{0};
      pointer m_data{nullptr};
    };

    struct resource_deleter
    {
    public:    
      template <typename Resource>
      resource_deleter(Resource res) : m_res{res}    
      {}

      void operator()(queue* ptr)
      {
        m_res.wait();     
        m_res.deallocate(ptr, camp::resources::MemoryAccess::Pinned); 
      } 
    private:
      resource_type m_res;
    };

  public:
    message_bus() : m_res{camp::resources::Host()}, 
      m_bus{m_res.allocate<queue>(1, camp::resources::MemoryAccess::Pinned),
            resource_deleter{m_res}} 
    {}

    template <typename Resource>
    message_bus(Resource res) : m_res{res}, 
      m_bus{new (m_res.allocate<queue>(1, camp::resources::MemoryAccess::Pinned)) queue{},
            resource_deleter{m_res}}
    {}

    template <typename Resource>
    message_bus(const size_type num_messages, Resource res) : message_bus{res}
    {
      reserve(num_messages);
    }

    ~message_bus()
    {
      reset();
    }

    // Copy ctor/operator
    message_bus(const message_bus&) = delete;
    message_bus& operator=(const message_bus&) = delete;

    // Move ctor/operator
    message_bus(message_bus&&) = default;
    message_bus& operator=(message_bus&&) = default;

    void reserve(size_type num_messages)
    {
      reset();
      m_bus->m_data = m_res.allocate<value_type>(num_messages,
        camp::resources::MemoryAccess::Pinned);
      m_bus->m_capacity = num_messages;
    }
  
    void reset()
    {
      // Verify that queue is not in use
      if (m_bus->m_data != nullptr) {
        m_res.wait();
        m_res.deallocate(m_bus->m_data,
          camp::resources::MemoryAccess::Pinned);
	m_bus->m_data = nullptr;
      }
      m_bus->m_capacity = 0;
      m_bus->m_size     = 0;
    }

    bool has_pending_messages()
    {
      return get_num_pending_messages() != 0;
    }

    size_type get_num_pending_messages()
    {
      m_res.wait();
      return std::min(m_bus->m_size, m_bus->m_capacity);
    }

    void clear_messages()
    {
      m_res.wait();
      m_bus->m_size = 0; 
    }

    // TODO: look into why this is returning an address that requires XNACK
    template <typename Policy>
    RAJA::messages::queue<queue, Policy> get_queue() const noexcept
    {
      return RAJA::messages::queue<queue, Policy>{m_bus.get()};
    } 

    iterator begin() noexcept 
    {
      return m_bus->m_data;
    }

    iterator end() noexcept 
    {
      return m_bus->m_data + get_num_pending_messages();
    }

  private:
    resource_type m_res; 
    std::unique_ptr<queue, resource_deleter> m_bus;
  };

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
    using callback_type = std::function<R(Args...)>;
    using msg_bus       = message_bus<message>;

  public:
    template <typename Callable>
    message_handler(const std::size_t num_messages, Callable c) 
      : m_bus{num_messages, camp::resources::Host()}, 
        m_callback{c}
    {}  

    template <typename Resource, typename Callable>
    message_handler(const std::size_t num_messages, Resource res, 
                    Callable c) 
      : m_bus{num_messages, res}, 
        m_callback{c}
    {}  

    ~message_handler() = default;

    // Doesn't support copying 
    message_handler(const message_handler&) = delete;
    message_handler& operator=(const message_handler&) = delete;

    // Move ctor/operator
    message_handler(message_handler&&) = default;
    message_handler& operator=(message_handler&&) = default;

    template <typename Policy>
    auto get_queue()
    {
      return m_bus.template get_queue<Policy>();
    } 

    void clear()
    {
      m_bus.clear_messages();
    }

    bool test_any()
    {
      return m_bus.has_pending_messages(); 
    }

    void wait_all()
    {
      if (test_any()) {
        for (const auto& msg: m_bus) {
          camp::apply(m_callback, msg);     
        }
        clear();
      }
    }

  private:
    msg_bus m_bus;
    callback_type m_callback;
  }; 
}

#endif /* RAJA_MESSAGES_HPP */
