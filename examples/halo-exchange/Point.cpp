//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstddef>
#include <cassert>
#include <list>

#include "Point.hpp"


// adds an item and returns the id of that item
size_t Point::addItem(double* var,
                      Order pack_order,
                      std::vector<int*>& pack_index_lists,
                      std::vector<int >& pack_index_list_lengths,
                      Order unpack_order,
                      std::vector<int*>& unpack_index_lists,
                      std::vector<int >& unpack_index_list_lengths)
{
  size_t id = m_items.size();
  m_items.emplace_back(
      new ItemNode(var,
                   pack_order,
                   pack_index_lists,
                   pack_index_list_lengths,
                   unpack_order,
                   unpack_index_lists,
                   unpack_index_list_lengths));
  return id;
}

// adds edge a -> b in graph
void Point::addDependency(size_t id_a, size_t id_b)
{
  assert(id_a < m_items.size());
  assert(id_b < m_items.size());

  std::unique_ptr<ItemNode>& node_a = m_items[id_a];
  std::unique_ptr<ItemNode>& node_b = m_items[id_b];

  node_a->dependent_items.emplace(id_b);
  node_b->num_parents += 1;
}

void Point::createSchedule()
{
  assert(!m_schedule);
  m_schedule.reset(new Schedule(m_my_rank));

  m_schedule->setDeterministicUnpackOrderingFlag(false);

  // perform a breadth first traversal
  // to visit the nodes in a valid order
  std::list<size_t> queue;

  // visit unordered items first
  for (size_t i = 0; i < m_items.size(); ++i) {
    if (m_items[i]->num_parents == 0u && m_items[i]->dependent_items.empty()) {
      queue.emplace_back(i);
    }
  }
  // then visit items with dependencies
  for (size_t i = 0; i < m_items.size(); ++i) {
    if (m_items[i]->num_parents == 0u && !m_items[i]->dependent_items.empty()) {
      queue.emplace_back(i);
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
    for (size_t o_id : node->dependent_items) {
      std::unique_ptr<ItemNode>& other_node = m_items[o_id];
      other_node->num += 1;
      if (other_node->num == other_node->num_parents) {
        other_node->num = 0;
        queue.emplace_back(o_id);
      }
    }

    // increase order requirements on transactions to get valid schedule
    // when items are ordered
    // changing unordered to reorderable prevents transactions
    // from different items being fused in an unordered fashion
    // (Schedule only has one fused phase for all transactions from all items)
    if (node->num_parents != 0 || !node->dependent_items.empty()) {

      if (node->item.getPackOrder() == Order::unordered) {
        node->item.setPackOrder(Order::reorderable);
      }
      if (node->item.getUnpackOrder() == Order::unordered) {
        node->item.setUnpackOrder(Order::reorderable);
      }

    }

    // Schedule does not support ordered packing transactions
    assert(node->item.getPackOrder() != Order::ordered);

    // Schedule supports ordered unpacking transactions
    if (node->item.getUnpackOrder() == Order::ordered) {
      m_schedule->setDeterministicUnpackOrderingFlag(true);
    }

    node->item.populate(*m_schedule);

  }

  assert(nodes_traversed == m_items.size());
}
