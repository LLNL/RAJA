/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkStorage.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WORKGROUP_WorkRunner_HPP
#define RAJA_PATTERN_WORKGROUP_WorkRunner_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>

#include "RAJA/policy/loop/policy.hpp"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"
#include "RAJA/policy/WorkGroup.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * A body and args holder for storing loops that are being executed in foralls
 */
template <typename LoopBody, typename ... Args>
struct HoldBodyArgs_base
{
  // NOTE: This constructor is disabled when body_in is not LoopBody
  // to avoid it conflicting with the copy and move constructors
  template < typename body_in,
      typename = typename std::enable_if<
        std::is_same<LoopBody, camp::decay<body_in>>::value>::type >
  HoldBodyArgs_base(body_in&& body, Args... args)
    : m_body(std::forward<body_in>(body))
    , m_arg_tuple(std::forward<Args>(args)...)
  { }

protected:
  LoopBody m_body;
  camp::tuple<Args...> m_arg_tuple;
};

/*!
 * A body and args holder for storing loops that are being executed in foralls
 * that run on the host
 */
template <typename LoopBody, typename index_type, typename ... Args>
struct HoldBodyArgs_host : HoldBodyArgs_base<LoopBody, Args...>
{
  using base = HoldBodyArgs_base<LoopBody, Args...>;
  using base::base;

  RAJA_INLINE void operator()(index_type i) const
  {
    invoke(i, camp::make_idx_seq_t<sizeof...(Args)>{});
  }

  template < camp::idx_t ... Is >
  RAJA_INLINE void invoke(index_type i, camp::idx_seq<Is...>) const
  {
    this->m_body(i, get<Is>(this->m_arg_tuple)...);
  }
};

/*!
 * A body and args holder for storing loops that are being executed in foralls
 * that run on the device
 */
template <typename LoopBody, typename index_type, typename ... Args>
struct HoldBodyArgs_device : HoldBodyArgs_base<LoopBody, Args...>
{
  using base = HoldBodyArgs_base<LoopBody, Args...>;
  using base::base;

  RAJA_DEVICE RAJA_INLINE void operator()(index_type i) const
  {
    invoke(i, camp::make_idx_seq_t<sizeof...(Args)>{});
  }

  template < camp::idx_t ... Is >
  RAJA_DEVICE RAJA_INLINE void invoke(index_type i, camp::idx_seq<Is...>) const
  {
    this->m_body(i, get<Is>(this->m_arg_tuple)...);
  }
};

/*!
 * A body and segment holder for storing loops that will be executed as foralls
 */
template <typename ExecutionPolicy, typename Segment_type, typename LoopBody,
          typename index_type, typename ... Args>
struct HoldForall
{
  using HoldBodyArgs = typename std::conditional<
      !type_traits::is_device_exec_policy<ExecutionPolicy>::value,
      HoldBodyArgs_host<LoopBody, index_type, Args...>,
      HoldBodyArgs_device<LoopBody, index_type, Args...> >::type;

  template < typename segment_in, typename body_in >
  HoldForall(segment_in&& segment, body_in&& body)
    : m_segment(std::forward<segment_in>(segment))
    , m_body(std::forward<body_in>(body))
  { }

  RAJA_INLINE void operator()(Args... args) const
  {
    wrap::forall(resources::get_resource<ExecutionPolicy>::type::get_default(),
                 ExecutionPolicy(),
                 m_segment,
                 HoldBodyArgs{m_body, std::forward<Args>(args)...});
  }

private:
  Segment_type m_segment;
  LoopBody m_body;
};


/*!
 * A class that handles running work in a work container
 */
template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunner;


/*!
 * Base class describing storage for ordered runners using forall
 */
template <typename FORALL_EXEC_POLICY,
          typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunnerForallOrdered_base
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using Allocator = ALLOCATOR_T;
  using index_type = INDEX_T;

  using forall_exec_policy = FORALL_EXEC_POLICY;
  using vtable_type = Vtable<void, Args...>;

  WorkRunnerForallOrdered_base() = default;

  WorkRunnerForallOrdered_base(WorkRunnerForallOrdered_base const&) = delete;
  WorkRunnerForallOrdered_base& operator=(WorkRunnerForallOrdered_base const&) = delete;

  WorkRunnerForallOrdered_base(WorkRunnerForallOrdered_base &&) = default;
  WorkRunnerForallOrdered_base& operator=(WorkRunnerForallOrdered_base &&) = default;

  // The type  that will hold the segment and loop body in work storage
  template < typename segment_type, typename loop_type >
  using holder_type = HoldForall<forall_exec_policy, segment_type, loop_type,
                                 index_type, Args...>;

  // The policy indicating where the call function is invoked
  // in this case the values are called on the host in a loop
  using vtable_exec_policy = RAJA::loop_work;

  // runner interfaces with storage to enqueue so the runner can get
  // information from the segment and loop at enqueue time
  template < typename WorkContainer, typename segment_T, typename loop_T >
  inline void enqueue(WorkContainer& storage, segment_T&& seg, loop_T&& loop)
  {
    using holder = holder_type<camp::decay<segment_T>, camp::decay<loop_T>>;

    storage.template emplace<holder>(
        get_Vtable<holder, vtable_type>(vtable_exec_policy{}),
        std::forward<segment_T>(seg), std::forward<loop_T>(loop));
  }

  // clear any state so ready to be destroyed or reused
  void clear()
  { }

  // no extra storage required here
  using per_run_storage = int;
};

/*!
 * Runs work in a storage container in order using forall
 */
template <typename FORALL_EXEC_POLICY,
          typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunnerForallOrdered
    : WorkRunnerForallOrdered_base<
      FORALL_EXEC_POLICY,
      EXEC_POLICY_T,
      ORDER_POLICY_T,
      ALLOCATOR_T,
      INDEX_T,
      Args...>
{
  using base = WorkRunnerForallOrdered_base<
      FORALL_EXEC_POLICY,
      EXEC_POLICY_T,
      ORDER_POLICY_T,
      ALLOCATOR_T,
      INDEX_T,
      Args...>;
  using base::base;

  // run the loops using forall in the order that they were enqueued
  template < typename WorkContainer >
  typename base::per_run_storage run(WorkContainer const& storage, Args... args) const
  {
    using value_type = typename WorkContainer::value_type;

    typename base::per_run_storage run_storage{};

    auto end = storage.end();
    for (auto iter = storage.begin(); iter != end; ++iter) {
      value_type::call(&*iter, args...);
    }

    return run_storage;
  }
};

/*!
 * Runs work in a storage container in reverse order using forall
 */
template <typename FORALL_EXEC_POLICY,
          typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunnerForallReverse
    : WorkRunnerForallOrdered_base<
      FORALL_EXEC_POLICY,
      EXEC_POLICY_T,
      ORDER_POLICY_T,
      ALLOCATOR_T,
      INDEX_T,
      Args...>
{
  using base = WorkRunnerForallOrdered_base<
      FORALL_EXEC_POLICY,
      EXEC_POLICY_T,
      ORDER_POLICY_T,
      ALLOCATOR_T,
      INDEX_T,
      Args...>;
  using base::base;

  // run the loops using forall in the reverse order to the order they were enqueued
  template < typename WorkContainer >
  typename base::per_run_storage run(WorkContainer const& storage, Args... args) const
  {
    using value_type = typename WorkContainer::value_type;

    typename base::per_run_storage run_storage{};

    auto begin = storage.begin();
    for (auto iter = storage.end(); iter != begin; --iter) {
      value_type::call(&*(iter-1), args...);
    }

    return run_storage;
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
