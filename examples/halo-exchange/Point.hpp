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
#include "Item.hpp"


struct Point
{
  Point() = default;

  Point(Point const&) = delete;
  Point& operator=(Point const&) = delete;

  ~Point() = default;

  // adds an item and returns the id of that item
  size_t addItem(double* var,
                 Order pack_order,
                 std::vector<int*>& pack_index_lists,
                 std::vector<int >& pack_index_list_lengths,
                 Order unpack_order,
                 std::vector<int*>& unpack_index_lists,
                 std::vector<int >& unpack_index_list_lengths);

  // adds edge a -> b in graph
  void addDependency(size_t id_a, size_t id_b);

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

  void clear()
  {
    destroySchedule();
    m_items.clear();
  }

private:
  struct ItemNode
  {
    template < typename... Args >
    ItemNode(Args&&... args)
      : item(std::forward<Args>(args)...)
    { }

    Item item;
    std::set<size_t> dependent_items;
    size_t num_parents = 0;
    size_t num = 0;
  };

  std::vector< std::unique_ptr<ItemNode> > m_items;
  std::unique_ptr< Schedule > m_schedule;
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
