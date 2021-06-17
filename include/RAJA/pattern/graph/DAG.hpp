/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::graph::DAG
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_graph_DAG_HPP
#define RAJA_pattern_graph_DAG_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <vector>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/graph/Node.hpp"
#include "RAJA/pattern/graph/Collection.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

namespace detail
{

template < typename GraphPolicy, typename GraphResource >
struct DAGExecBase
{
  static_assert(type_traits::is_execution_policy<GraphPolicy>::value,
                "GraphPolicy is not a policy");
  static_assert(pattern_is<GraphPolicy, Pattern::graph>::value,
                "GraphPolicy is not a graph policy");
  static_assert(type_traits::is_resource<GraphResource>::value,
                "GraphResource is not a resource");
};

} // namespace detail

template < typename GraphPolicy, typename GraphResource >
struct DAGExec;

struct DAG
{
  struct GenericNodeView;

  template < typename node_type >
  struct NodeView
  {
    using args_type = typename node_type::args_type;

    id_type id = invalid_id;
    node_type* node = nullptr;

    NodeView() = default;

    NodeView(id_type id_, node_type* node_) noexcept
      : id(id_)
      , node(node_)
    {
    }

    NodeView(NodeView const&) = default;
    NodeView(NodeView&&) = default;

    NodeView& operator=(NodeView const&) = default;
    NodeView& operator=(NodeView&&) = default;

    ~NodeView() = default;

    bool try_reset(args_type const& args) noexcept
    {
      if (*this) {
        (*node) = args;
        return true;
      }
      return false;
    }
    bool try_reset(args_type&& args) noexcept
    {
      if (*this) {
        (*node) = std::move(args);
        return true;
      }
      return false;
    }

    void reset(args_type const& args)
    {
      if (!try_reset(args)) {
        throw std::runtime_error("NodeView::reset failed, no node to reset");
      }
    }
    void reset(args_type&& args)
    {
      if (!try_reset(std::move(args))) {
        throw std::runtime_error("NodeView::reset failed, no node to reset");
      }
    }

    node_type& operator*() const noexcept
    {
      return *node;
    }

    node_type* operator->() const noexcept
    {
      return node;
    }

    operator GenericNodeView() const noexcept
    {
      return {id, node};
    }

    operator id_type() const noexcept
    {
      return id;
    }

    explicit operator bool() const noexcept
    {
      return id != invalid_id && node != nullptr;
    }
  };

  struct GenericNodeView
  {
    id_type id = invalid_id;
    detail::NodeData* node = nullptr;

    GenericNodeView() = default;

    GenericNodeView(id_type id_, detail::NodeData* node_) noexcept
      : id(id_)
      , node(node_)
    {
    }

    GenericNodeView(GenericNodeView const&) = default;
    GenericNodeView(GenericNodeView&&) = default;

    GenericNodeView& operator=(GenericNodeView const&) = default;
    GenericNodeView& operator=(GenericNodeView&&) = default;

    ~GenericNodeView() = default;

    template < typename node_type >
    NodeView<node_type> try_get() const noexcept
    {
      node_type* typed_node = nullptr;
      if (*this) {
        typed_node = dynamic_cast<node_type*>(node);
      }
      if (typed_node != nullptr) {
        return {id, typed_node};
      } else {
        return {};
      }
    }

    template < typename node_type >
    NodeView<node_type> get() const
    {
      NodeView<node_type> typed_node_view = try_get<node_type>();

      if (*this && !typed_node_view) {
        throw std::runtime_error("GenericNodeView::get failed to convert to node_type");
      }

      return typed_node_view;
    }

    template < typename node_args >
    bool try_reset(node_args&& args) noexcept
    {
      if (*this) {
        using node_type = typename camp::decay<node_args>::node_type;
        NodeView<node_type> typed_node_view = try_get<node_type>();
        if (typed_node_view) {
          return typed_node_view.try_reset(std::forward<node_args>(args));
        }
      }
      return false;
    }

    template < typename node_args >
    void reset(node_args&& args)
    {
      if (*this) {
        using node_type = typename camp::decay<node_args>::node_type;
        NodeView<node_type> typed_node_view = get<node_type>();
        typed_node_view.reset(std::forward<node_args>(args));
      } else {
        throw std::runtime_error("GenericNodeView::reset failed, this has no node");
      }
    }

    operator id_type() const noexcept
    {
      return id;
    }

    explicit operator bool() const noexcept
    {
      return id != invalid_id && node != nullptr;
    }
  };

  template < typename node_args >
  using node_type = typename camp::decay<node_args>::node_type;

  template < typename node_args >
  using node_view = NodeView< node_type<node_args> >;


  struct GenericCollectionView;

  template < typename collection_type >
  struct CollectionView
  {
    using args_type = typename collection_type::args_type;

    id_type id = invalid_id;
    collection_type* collection = nullptr;

    CollectionView() = default;

    CollectionView(id_type id_, collection_type* collection_) noexcept
      : id(id_)
      , collection(collection_)
    {
    }

    CollectionView(CollectionView const&) = default;
    CollectionView(CollectionView&&) = default;

    CollectionView& operator=(CollectionView const&) = default;
    CollectionView& operator=(CollectionView&&) = default;

    ~CollectionView() = default;

    collection_type& operator*() const noexcept
    {
      return *collection;
    }

    collection_type* operator->() const noexcept
    {
      return collection;
    }

    operator GenericCollectionView() const noexcept
    {
      return {id, collection};
    }

    operator id_type() const noexcept
    {
      return id;
    }

    explicit operator bool() const noexcept
    {
      return id != invalid_id && collection != nullptr;
    }
  };

  struct GenericCollectionView
  {
    id_type id = invalid_id;
    detail::Collection* collection = nullptr;

    GenericCollectionView() = default;

    GenericCollectionView(id_type id_, detail::Collection* collection_) noexcept
      : id(id_)
      , collection(collection_)
    {
    }

    GenericCollectionView(GenericCollectionView const&) = default;
    GenericCollectionView(GenericCollectionView&&) = default;

    GenericCollectionView& operator=(GenericCollectionView const&) = default;
    GenericCollectionView& operator=(GenericCollectionView&&) = default;

    ~GenericCollectionView() = default;

    template < typename collection_type >
    CollectionView<collection_type> try_get() const noexcept
    {
      collection_type* typed_collection = nullptr;
      if (*this) {
        typed_collection = dynamic_cast<collection_type*>(collection);
      }
      if (typed_collection != nullptr) {
        return {id, typed_collection};
      } else {
        return {};
      }
    }

    template < typename collection_type >
    CollectionView<collection_type> get() const
    {
      CollectionView<collection_type> typed_collection_view = try_get<collection_type>();

      if (*this && !typed_collection_view) {
        throw std::runtime_error("GenericCollectionView::get failed to convert to collection_type");
      }

      return typed_collection_view;
    }

    operator id_type() const noexcept
    {
      return id;
    }

    explicit operator bool() const noexcept
    {
      return id != invalid_id && collection != nullptr;
    }
  };

  template < typename collection_args >
  using collection_type = typename ::camp::decay<collection_args>::collection_type;

  template < typename collection_args >
  using collection_view = CollectionView< collection_type<collection_args> >;


  DAG() = default;

  bool empty() const
  {
    return m_node_connections.empty();
  }

  template < typename collection_args >
  collection_view<collection_args> add_collection(collection_args&& args)
  {
    auto collection = new collection_type<collection_args>(invalid_id, std::move(args));
    // store collection in unique_ptr in container, get id
    id_type collection_id = insert_collection(collection);
    return {collection_id, collection};
  }

  template < typename collection_type, typename node_args >
  node_view<node_args> add_collection_node(
      CollectionView<collection_type> const& collection_view,
      node_args&& args)
  {
    id_type collection_inner_id = invalid_id;
    auto node = new node_type<node_args>(
        *collection_view, collection_inner_id,
        std::forward<node_args>(args));
    // store node in unique_ptr in container, get id
    id_type node_id = insert_collection_node(node, collection_view.id, collection_inner_id);
    return {node_id, node};
  }

  template < typename node_args >
  node_view<node_args> add_node(node_args&& args)
  {
    auto node = new node_type<node_args>(std::forward<node_args>(args));
    // store node in unique_ptr in container, get id
    id_type node_id = insert_node(node);
    return {node_id, node};
  }

  void add_edge(id_type id_a, id_type id_b)
  {
#if defined(RAJA_BOUNDS_CHECK_INTERNAL)
    if(id_a >= m_node_connections.size()) {
      std::string err;
      err += "Error! DAG::add_edge id_a ";
      err += std::to_string(id_a);
      err += " is not valid.";
      throw std::runtime_error(std::move(err));
    }
    if(id_b >= m_node_connections.size()) {
      std::string err;
      err += "Error! DAG::add_edge id_b ";
      err += std::to_string(id_b);
      err += " is not valid.";
      throw std::runtime_error(std::move(err));
    }
#endif
    m_node_connections[id_a].add_child(m_node_connections[id_b]);
  }

  template < typename GraphPolicy, typename GraphResource >
  DAGExec<GraphPolicy, GraphResource> instantiate()
  {
    return {*this};
  }

  void clear()
  {
    m_node_connections.clear();
    m_node_data = std::make_shared<node_data_container>();
  }

  ~DAG() = default;

private:
  template < typename, typename >
  friend struct DAGExec;

  struct node_data_container
  {
    std::vector<std::unique_ptr<detail::Collection>> collections;
    std::vector<std::unique_ptr<detail::NodeData>> node_data;
  };

  std::vector<detail::NodeConnections> m_node_connections;
  std::shared_ptr<node_data_container> m_node_data = std::make_shared<node_data_container>();

  id_type insert_collection(detail::Collection* collection)
  {
    id_type collection_id = m_node_data->collections.size();
    m_node_data->collections.emplace_back(collection);
    collection->set_my_id(collection_id);
    return collection_id;
  }

  id_type insert_collection_node(detail::NodeData* node_data,
                                 id_type collection_id,
                                 id_type collection_inner_id)
  {
    id_type node_id = m_node_data->node_data.size();
    m_node_data->node_data.emplace_back(node_data);
    m_node_connections.emplace_back(node_id, collection_id, collection_inner_id);
    return node_id;
  }

  id_type insert_node(detail::NodeData* node_data)
  {
    id_type node_id = m_node_data->node_data.size();
    m_node_data->node_data.emplace_back(node_data);
    m_node_connections.emplace_back(node_id);
    return node_id;
  }

  // Depth first traversal of nodes in an order consistent with the DAG,
  // calling enter_func when traversing a node before traversing any of the
  // node's children and calling exit_func after examining all of the node's
  // children. NOTE that exit_function is not necessarily called after
  // exit_function is called on each of the node's children. NOTE that a
  // node is not used again after exit_function is called on it.
  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_depth_first_traversal(Examine_Func&& examine_func,
                                     Enter_Func&& enter_func,
                                     Exit_Func&& exit_func)
  {
    for (detail::NodeConnections& child : m_node_connections)
    {
      if (child.m_parents.size() == 0) {
        std::forward<Examine_Func>(examine_func)(child);
        child.forward_depth_first_traversal(
            m_node_connections.data(),
            std::forward<Examine_Func>(examine_func),
            std::forward<Enter_Func>(enter_func),
            std::forward<Exit_Func>(exit_func));
      }
    }
  }

  // Breadth first traversal of nodes in an order consistent with the DAG,
  // calling enter_func when traversing a node before traversing any of the
  // node's children and calling exit_func after examining all of the node's
  // children. NOTE that exit_function is called before exit_function is
  // called on each of the node's children. NOTE that a node is not used
  // again after exit_function is called on it.
  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_breadth_first_traversal(Examine_Func&& examine_func,
                                       Enter_Func&& enter_func,
                                       Exit_Func&& exit_func)
  {
    std::list<detail::NodeConnections*> queue;
    for (detail::NodeConnections& child : m_node_connections)
    {
      if (child.m_parents.size() == 0) {
        std::forward<Examine_Func>(examine_func)(child);
        queue.emplace_back(&child);
      }
    }
    while (!queue.empty())
    {
      detail::NodeConnections* child = queue.front();
      queue.pop_front();
      child->forward_breadth_first_traversal(
          queue, m_node_connections.data(),
          std::forward<Examine_Func>(examine_func),
          std::forward<Enter_Func>(enter_func),
          std::forward<Exit_Func>(exit_func));
    }
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
