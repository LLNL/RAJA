//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstddef>
#include <cassert>
#include <list>
#include <unordered_set>

#include "Point.hpp"


// adds an item and returns the id of that item
size_t Point::addItem(double* var,
                      Order pack_transaction_order,
                      std::vector<int*>& pack_index_lists,
                      std::vector<int >& pack_index_list_lengths,
                      Order unpack_transaction_order,
                      std::vector<int*>& unpack_index_lists,
                      std::vector<int >& unpack_index_list_lengths)
{
  size_t id = m_items.size();
  m_items.emplace_back(
      new ItemNode(var,
                   pack_transaction_order,
                   pack_index_lists,
                   pack_index_list_lengths,
                   unpack_transaction_order,
                   unpack_index_lists,
                   unpack_index_list_lengths));
  return id;
}

// adds edge a -> b in graph
void Point::addDependency(size_t id_a, size_t id_b)
{
  addPackDependency(id_a, id_b);
  addUnpackDependency(id_a, id_b);
}
void Point::addPackDependency(size_t id_a, size_t id_b)
{
  assert(id_a < m_items.size());
  assert(id_b < m_items.size());

  std::unique_ptr<ItemNode>& node_a = m_items[id_a];
  std::unique_ptr<ItemNode>& node_b = m_items[id_b];

  node_a->pack_connectivity.dependent_items.emplace(id_b);
  node_b->pack_connectivity.num_parents += 1;
}
void Point::addUnpackDependency(size_t id_a, size_t id_b)
{
  assert(id_a < m_items.size());
  assert(id_b < m_items.size());

  std::unique_ptr<ItemNode>& node_a = m_items[id_a];
  std::unique_ptr<ItemNode>& node_b = m_items[id_b];

  node_a->unpack_connectivity.dependent_items.emplace(id_b);
  node_b->unpack_connectivity.num_parents += 1;
}

void Point::createSchedule()
{
  assert(!m_schedule);
  m_schedule.reset(new Schedule(m_my_rank));

  m_schedule->setDeterministicUnpackOrderingFlag(false);

  bool allow_pack_items_unordered = true;
  bool allow_unpack_items_unordered = true;

  // perform a breadth first traversal
  // to visit the nodes in a valid order
  using queue_type = std::list<size_t>;
  using queue_iterator = typename queue_type::iterator;
  std::list<size_t> queue;
  queue_iterator first_with_dependencies = queue.end();

  for (size_t i = 0; i < m_items.size(); ++i) {
    if (m_items[i]->pack_connectivity.num_parents == 0u &&
        m_items[i]->unpack_connectivity.num_parents == 0u) {
      if (m_items[i]->pack_connectivity.dependent_items.empty() &&
          m_items[i]->unpack_connectivity.dependent_items.empty()) {
        // visit unordered items before any with dependencies
        queue.emplace(first_with_dependencies, i);
      } else {
        // then visit items with dependencies
        queue_iterator it = queue.emplace(queue.end(), i);
        if (first_with_dependencies == queue.end()) {
          first_with_dependencies = it;
        }
      }
    }
  }

  size_t nodes_traversed = 0;

  while (!queue.empty()) {

    nodes_traversed += 1;

    size_t id = queue.front();
    queue.pop_front();

    std::unique_ptr<ItemNode>& node = m_items[id];

    // examine dependent nodes
    // adding any that are ready to go to the queue
    for (size_t o_id : node->pack_connectivity.dependent_items) {
      std::unique_ptr<ItemNode>& other_node = m_items[o_id];
      other_node->pack_connectivity.num += 1;
      if (other_node->pack_connectivity.num == other_node->pack_connectivity.num_parents &&
          other_node->unpack_connectivity.num == other_node->unpack_connectivity.num_parents) {
        other_node->pack_connectivity.num = 0;
        other_node->unpack_connectivity.num = 0;
        queue.emplace_back(o_id);
      }
    }
    for (size_t o_id : node->unpack_connectivity.dependent_items) {
      std::unique_ptr<ItemNode>& other_node = m_items[o_id];
      other_node->unpack_connectivity.num += 1;
      if (other_node->pack_connectivity.num == other_node->pack_connectivity.num_parents &&
          other_node->unpack_connectivity.num == other_node->unpack_connectivity.num_parents) {
        other_node->pack_connectivity.num = 0;
        other_node->unpack_connectivity.num = 0;
        queue.emplace_back(o_id);
      }
    }

    // increase order requirements on transactions to get valid schedule
    // when items are ordered
    // changing unordered to reorderable prevents transactions
    // from different items being fused in an unordered fashion
    // (Schedule only has one fused phase for all transactions from all items)

    // fuse pack transactions that don't depend on anything
    if (allow_pack_items_unordered) {
      allow_pack_items_unordered = node->pack_connectivity.num_parents == 0;
    }
    if (!allow_pack_items_unordered &&
        node->item.getPackTransactionOrder() == Order::unordered) {
      node->item.setPackTransactionOrder(Order::reorderable);
    }

    // fuse unpack transactions that don't depend on anything and don't have dependencies
    if (allow_unpack_items_unordered) {
      allow_unpack_items_unordered = node->unpack_connectivity.num_parents == 0 &&
                                     node->unpack_connectivity.dependent_items.empty();
    }
    if (!allow_unpack_items_unordered &&
        node->item.getUnpackTransactionOrder() == Order::unordered) {
      node->item.setUnpackTransactionOrder(Order::reorderable);
    }

    // Schedule does not support ordered packing transactions
    assert(node->item.getPackTransactionOrder() != Order::ordered &&
           "Schedule does not support ordered packing transactions");

    // Schedule supports ordered unpacking transactions
    if (node->item.getUnpackTransactionOrder() == Order::ordered) {
      m_schedule->setDeterministicUnpackOrderingFlag(true);
    }

    node->item.populate(*m_schedule);

  }

  assert(nodes_traversed == m_items.size() &&
         "Cycle in items dependency graph or pack and unpack are not compatible (the edges of one must be a subset of the edges of the other)");
}


void Point::createGraphSchedule()
{
  assert(!m_graphSchedule);
  m_graphSchedule.reset(new GraphSchedule(m_my_rank));

  m_graphSchedule->setDeterministicUnpackOrderingFlag(false);

  bool allow_pack_items_unordered = true;
  bool allow_unpack_items_unordered = true;

  // perform a breadth first traversal
  // to visit the nodes in a valid order
  using queue_type = std::list<size_t>;
  using queue_iterator = typename queue_type::iterator;
  std::list<size_t> queue;
  queue_iterator first_with_dependencies = queue.end();

  for (size_t i = 0; i < m_items.size(); ++i) {
    if (m_items[i]->pack_connectivity.num_parents == 0u &&
        m_items[i]->unpack_connectivity.num_parents == 0u) {
      if (m_items[i]->pack_connectivity.dependent_items.empty() &&
          m_items[i]->unpack_connectivity.dependent_items.empty()) {
        // visit unordered items before any with dependencies
        queue.emplace(first_with_dependencies, i);
      } else {
        // then visit items with dependencies
        queue_iterator it = queue.emplace(queue.end(), i);
        if (first_with_dependencies == queue.end()) {
          first_with_dependencies = it;
        }
      }
    }
  }

  size_t nodes_traversed = 0;

  while (!queue.empty()) {

    nodes_traversed += 1;

    size_t id = queue.front();
    queue.pop_front();

    std::unique_ptr<ItemNode>& node = m_items[id];

    // examine dependent nodes
    // adding any that are ready to go to the queue
    for (size_t o_id : node->pack_connectivity.dependent_items) {
      std::unique_ptr<ItemNode>& other_node = m_items[o_id];
      other_node->pack_connectivity.num += 1;
      if (other_node->pack_connectivity.num == other_node->pack_connectivity.num_parents &&
          other_node->unpack_connectivity.num == other_node->unpack_connectivity.num_parents) {
        other_node->pack_connectivity.num = 0;
        other_node->unpack_connectivity.num = 0;
        queue.emplace_back(o_id);
      }
    }
    for (size_t o_id : node->unpack_connectivity.dependent_items) {
      std::unique_ptr<ItemNode>& other_node = m_items[o_id];
      other_node->unpack_connectivity.num += 1;
      if (other_node->pack_connectivity.num == other_node->pack_connectivity.num_parents &&
          other_node->unpack_connectivity.num == other_node->unpack_connectivity.num_parents) {
        other_node->pack_connectivity.num = 0;
        other_node->unpack_connectivity.num = 0;
        queue.emplace_back(o_id);
      }
    }

    // increase order requirements on transactions to get valid schedule
    // when items are ordered
    // changing unordered to reorderable prevents transactions
    // from different items being fused in an unordered fashion
    // (GraphSchedule only has one fused phase for all transactions from all items)

    // fuse pack transactions that don't depend on anything
    if (allow_pack_items_unordered) {
      allow_pack_items_unordered = node->pack_connectivity.num_parents == 0;
    }
    if (!allow_pack_items_unordered &&
        node->item.getPackTransactionOrder() == Order::unordered) {
      node->item.setPackTransactionOrder(Order::reorderable);
    }

    // fuse unpack transactions that don't depend on anything and don't have dependencies
    if (allow_unpack_items_unordered) {
      allow_unpack_items_unordered = node->unpack_connectivity.num_parents == 0 &&
                                     node->unpack_connectivity.dependent_items.empty();
    }
    if (!allow_unpack_items_unordered &&
        node->item.getUnpackTransactionOrder() == Order::unordered) {
      node->item.setUnpackTransactionOrder(Order::reorderable);
    }

    // GraphSchedule does not support ordered packing transactions
    assert(node->item.getPackTransactionOrder() != Order::ordered &&
           "GraphSchedule does not support ordered packing transactions");

    // GraphSchedule supports ordered unpacking transactions
    if (node->item.getUnpackTransactionOrder() == Order::ordered) {
      m_graphSchedule->setDeterministicUnpackOrderingFlag(true);
    }

    node->item.populate(*m_graphSchedule);

  }

  assert(nodes_traversed == m_items.size() &&
         "Cycle in items dependency graph or pack and unpack are not compatible (the edges of one must be a subset of the edges of the other)");
}
