/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::graph::Node
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_graph_WorkGroupCollection_HPP
#define RAJA_pattern_graph_WorkGroupCollection_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>
#include <vector>

#include "RAJA/pattern/WorkGroup/WorkStorage.hpp"
#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"

#include "RAJA/pattern/graph/Collection.hpp"

#include "RAJA/util/camp_aliases.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct FusibleForallNode;

template < typename GraphPolicy, typename GraphResource >
struct DAGExec;

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
struct WorkGroupCollection;

namespace detail
{

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
struct WorkGroupCollectionArgs : ::RAJA::expt::graph::detail::CollectionArgs
{
  using collection_type = ::RAJA::expt::graph::WorkGroupCollection<
                                                   EXEC_POLICY_T,
                                                   ORDER_POLICY_T,
                                                   INDEX_T,
                                                   EXTRA_ARGS_T,
                                                   ALLOCATOR_T >;

  WorkGroupCollectionArgs(ALLOCATOR_T const& aloc)
    : m_aloc(aloc)
  {
  }

  ALLOCATOR_T m_aloc;
};

// based on WorkStorage<RAJA::array_of_pointers
// but allows external storage of items
template < typename ALLOCATOR_T, typename Vtable_T >
struct WorkCollectionStorage
{
  using vtable_type = Vtable_T;

  template < typename holder >
  using true_value_type = ::RAJA::detail::WorkStruct<sizeof(holder), vtable_type>;

  using value_type = ::RAJA::detail::GenericWorkStruct<vtable_type>;
  using allocator_type = ALLOCATOR_T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  // struct used in storage vector to retain pointer and allocation size
  struct pointer_and_size
  {
    pointer ptr;
    size_type size;
  };

private:
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using pointer_and_size_allocator =
      typename allocator_traits_type::template rebind_alloc<pointer_and_size>;
  static_assert(std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkCollectionStorage expects an allocator for 'char's.");

public:

  // iterator base class for accessing stored WorkStructs outside of the container
  struct const_iterator_base
  {
    using value_type = const typename WorkCollectionStorage::value_type;
    using pointer = typename WorkCollectionStorage::const_pointer;
    using reference = typename WorkCollectionStorage::const_reference;
    using difference_type = typename WorkCollectionStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator_base(const pointer_and_size* ptrptr)
      : m_ptrptr(ptrptr)
    { }

    RAJA_HOST_DEVICE reference operator*() const
    {
      return *(m_ptrptr->ptr);
    }

    RAJA_HOST_DEVICE const_iterator_base& operator+=(difference_type n)
    {
      m_ptrptr += n;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator_base const& lhs_iter, const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_ptrptr - rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator_base const& lhs_iter, const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_ptrptr == rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator_base const& lhs_iter, const_iterator_base const& rhs_iter)
    {
      return lhs_iter.m_ptrptr < rhs_iter.m_ptrptr;
    }

  private:
    const pointer_and_size* m_ptrptr;
  };

  using const_iterator = ::RAJA::detail::random_access_iterator<const_iterator_base>;

  WorkCollectionStorage() = delete;

  WorkCollectionStorage(WorkCollectionStorage const&) = delete;
  WorkCollectionStorage& operator=(WorkCollectionStorage const&) = delete;
  WorkCollectionStorage(WorkCollectionStorage&& rhs) = delete;
  WorkCollectionStorage& operator=(WorkCollectionStorage&& rhs) = delete;

  explicit WorkCollectionStorage(allocator_type const& aloc)
    : m_vec(0, aloc)
  { }

  ~WorkCollectionStorage() = default;

  // reserve may be used to allocate enough memory to store num_loops
  // and loop_storage_size is ignored in this version because each
  // object has its own allocation
  void reserve(size_type num_loops)
  {
    m_vec.reserve(num_loops);
  }

  // number of loops stored
  size_type size() const
  {
    return m_vec.size();
  }

  const_iterator begin() const
  {
    return const_iterator(m_vec.begin());
  }

  const_iterator end() const
  {
    return const_iterator(m_vec.end());
  }

  void insert(pointer_and_size const& value_and_size_ptr)
  {
    m_vec.emplace_back(value_and_size_ptr);
  }
  ///
  void insert(pointer_and_size&& value_and_size_ptr)
  {
    m_vec.emplace_back(std::move(value_and_size_ptr));
  }

  void clear()
  {
    m_vec.clear();
  }

  // allocate and construct value in storage
  template < typename holder, typename ... holder_ctor_args >
  static pointer_and_size create_value(allocator_type& aloc,
                                       const vtable_type* vtable,
                                       holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    pointer value_ptr = reinterpret_cast<pointer>(
        allocator_traits_type::allocate(aloc, value_size));

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);

    return pointer_and_size{value_ptr, value_size};
  }

  // destroy and deallocate value
  static void destroy_value(allocator_type& aloc,
                            pointer_and_size value_and_size_ptr)
  {
    value_type::destroy(value_and_size_ptr.ptr);
    allocator_traits_type::deallocate(aloc,
        reinterpret_cast<char*>(value_and_size_ptr.ptr), value_and_size_ptr.size);
  }

private:
  RAJAVec<pointer_and_size,
          pointer_and_size_allocator> m_vec;
};

}  // namespace detail


template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
::RAJA::expt::graph::detail::WorkGroupCollectionArgs<EXEC_POLICY_T,
                                                     ORDER_POLICY_T,
                                                     INDEX_T,
                                                     EXTRA_ARGS_T,
                                                     ALLOCATOR_T>
WorkGroup(ALLOCATOR_T const& aloc)
{
  return ::RAJA::expt::graph::detail::WorkGroupCollectionArgs<EXEC_POLICY_T,
                                                              ORDER_POLICY_T,
                                                              INDEX_T,
                                                              EXTRA_ARGS_T,
                                                              ALLOCATOR_T>(
      aloc);
}


template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename INDEX_T,
          typename ... Args,
          typename ALLOCATOR_T>
struct WorkGroupCollection<EXEC_POLICY_T,
                           ORDER_POLICY_T,
                           INDEX_T,
                           ::RAJA::xargs<Args...>,
                           ALLOCATOR_T>
    : ::RAJA::expt::graph::detail::Collection
{
  using base = ::RAJA::expt::graph::detail::Collection;

  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = ::RAJA::array_of_pointers;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;
  using resource = typename ::RAJA::resources::get_resource<exec_policy>::type;

  using args_type = ::RAJA::expt::graph::detail::WorkGroupCollectionArgs<
                                                     exec_policy,
                                                     order_policy,
                                                     index_type,
                                                     xarg_type,
                                                     Allocator>;

  WorkGroupCollection() = delete;

  WorkGroupCollection(WorkGroupCollection const&) = delete;
  WorkGroupCollection(WorkGroupCollection&&) = delete;

  WorkGroupCollection& operator=(WorkGroupCollection const&) = delete;
  WorkGroupCollection& operator=(WorkGroupCollection&&) = delete;

  WorkGroupCollection(size_t id, args_type const& args)
    : base(id)
    , m_aloc(args.m_aloc)
  {
  }

  virtual ~WorkGroupCollection()
  {
    for (pointer_and_size& value_and_size_ptr : m_values) {
      storage_type::destroy_value(m_aloc, value_and_size_ptr);
    }
  }

  detail::CollectionNodeData* newExecNode() override
  {
    return new ExecNode(this);
  }

  // make_FusedNode()

protected:
  template < typename, typename, typename >
  friend struct FusibleForallNode;

  template < typename, typename >
  friend struct DAGExec;

  using workrunner_type = ::RAJA::detail::WorkRunner<exec_policy,
                                                     order_policy,
                                                     Allocator,
                                                     index_type,
                                                     Args...>;

  // The policy indicating where the call function is invoked
  using vtable_exec_policy = typename workrunner_type::vtable_exec_policy;
  // cuda thinks this is incomplete generating device code
  using vtable_type = typename workrunner_type::vtable_type;
  // using vtable_type = RAJA::detail::Vtable<exec_policy, Args...>;

  template < typename Container, typename LoopBody >
  using runner_holder_type = typename workrunner_type::template holder_type<Container, LoopBody>;
  template < typename Container, typename LoopBody >
  using runner_caller_type = typename workrunner_type::template caller_type<Container, LoopBody>;

  using storage_type = ::RAJA::expt::graph::detail::WorkCollectionStorage<Allocator,
                                                                          vtable_type>;

  using pointer_and_size = typename storage_type::pointer_and_size;
  using value_type = typename storage_type::value_type;

  struct ExecNode : detail::CollectionNodeData
  {
    ExecNode(WorkGroupCollection* collection)
      : m_collection(collection)
      , m_storage(collection->m_aloc)
      , m_runner()
      , m_run_storage()
      , m_args(Args{}...)
    {
    }

    void enqueue(detail::NodeData* node,
                 id_type collection_inner_id) override
    {
      m_storage.insert(m_collection->m_values[collection_inner_id]);
      m_runner.addLoopIterations(node->get_num_iterations());
    }

    size_t get_num_iterations() const override
    {
      return 0;
    }

    void exec() override
    {
      exec_impl(camp::make_idx_seq_t<sizeof...(Args)>{});
    }

  private:
    using per_run_storage = typename workrunner_type::per_run_storage;

    WorkGroupCollection* m_collection;
    storage_type         m_storage;
    workrunner_type      m_runner;
    per_run_storage      m_run_storage;
    RAJA::tuple<Args...> m_args;

    template < camp::idx_t ... Is >
    void exec_impl(camp::idx_seq<Is...>)
    {
      resource r = resource::get_default();

      m_run_storage = m_runner.run(m_storage, r, RAJA::get<Is>(m_args)...);

      r.wait();
    }
  };


  std::vector<pointer_and_size> m_values;
  Allocator m_aloc;


  // Create items with storage for use in storage_type.
  // Note that this object owns the storage not instances of storage_type, so
  // where storage_type is created take care to avoid double freeing the items.
  template < typename Container, typename LoopBody >
  std::pair<runner_holder_type<::camp::decay<Container>, ::camp::decay<LoopBody>>*,
            id_type>
  emplace(Container&& c, LoopBody&& body)
  {
    using holder = runner_caller_type<::camp::decay<Container>, ::camp::decay<LoopBody>>;

    const vtable_type* vtable = ::RAJA::detail::get_Vtable<holder, vtable_type>(
        vtable_exec_policy{});

    pointer_and_size value = storage_type::template create_value<holder>(
        m_aloc, vtable, std::forward<Container>(c), std::forward<LoopBody>(body));

    id_type inner_id = m_values.size();
    m_values.emplace_back(std::move(value));

    m_num_nodes = m_values.size();

    return {value_type::template get_holder<holder>(m_values.back().ptr),
            inner_id};
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
