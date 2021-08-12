//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_POINT_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_POINT_HPP

#include <cstddef>
#include <cassert>
#include <memory>
#include <vector>
#include <set>

#include "Schedule.hpp"
#include "GraphSchedule.hpp"
#include "Item.hpp"


struct Point
{
  using item_id_type = size_t;

  Point() = default;

  Point(Point const&) = delete;
  Point& operator=(Point const&) = delete;

  ~Point() = default;

  // adds an item and returns the id of that item
  size_t addItem(double* var,
                 Order pack_transaction_order,
                 std::vector<int*>& pack_index_lists,
                 std::vector<int >& pack_index_list_lengths,
                 Order unpack_transaction_order,
                 std::vector<int*>& unpack_index_lists,
                 std::vector<int >& unpack_index_list_lengths,
                 TransactionType transaction_type);

  // adds edge a -> b in graph
  void addDependency(size_t id_a, size_t id_b);
  void addPackDependency(size_t id_a, size_t id_b);
  void addUnpackDependency(size_t id_a, size_t id_b);

  void createSchedule();

  void destroySchedule()
  {
    m_schedule.reset();
  }

  Schedule& getSchedule()
  {
    assert(m_schedule);
    return *m_schedule;
  }

  void createGraphSchedule();

  void destroyGraphSchedule()
  {
    m_graphSchedule.reset();
  }

  GraphSchedule& getGraphSchedule()
  {
    assert(m_graphSchedule);
    return *m_graphSchedule;
  }

  void clear()
  {
    destroySchedule();
    destroyGraphSchedule();
    m_items.clear();
  }

private:
  struct Connectivity
  {
    std::set<size_t> dependent_items;
    size_t num_parents = 0;
    size_t num = 0;
  };
  struct ItemNode
  {
    template < typename... Args >
    ItemNode(Args&&... args)
      : item(std::forward<Args>(args)...)
    { }

    Item item;
    Connectivity pack_connectivity;
    Connectivity unpack_connectivity;
  };

  std::vector< std::unique_ptr<ItemNode> > m_items;
  std::unique_ptr< Schedule > m_schedule;
  std::unique_ptr< GraphSchedule > m_graphSchedule;
  int m_my_rank = 0;
};


// point
//   has items ordered in some way (implicit ordered -> graph skeleton)
//     knows ordering between transactions
//       packing reorderable
//       unpacking reorderable or ordered
//     make transactions
//   has
//   has schedule
//     has transactions
//
// orderings
//   ordered
//   reorderable
//   unordered
//


#endif // RAJA_EXAMPLES_HALOEXCHANGE_POINT_HPP
