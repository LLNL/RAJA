//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_ITEM_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_ITEM_HPP

#include <cstddef>
#include <vector>

#include "Schedule.hpp"


enum struct Order
{
  ordered,
  reorderable,
  unordered
};


struct Item
{
  Item(Order pack_order, Order unpack_order,
       double* var,
       std::vector<int*>& pack_index_lists,
       std::vector<int >& pack_index_list_lengths,
       std::vector<int*>& unpack_index_lists,
       std::vector<int >& unpack_index_list_lengths);

  Item(Item const&) = delete;
  Item& operator=(Item const&) = delete;

  ~Item() = default;

  void populate(Schedule& schedule);

private:
  Order m_pack_order;
  Order m_unpack_order;
  double* m_var;
  std::vector<int*> m_pack_index_lists;
  std::vector<int > m_pack_index_list_lengths;
  std::vector<int*> m_unpack_index_lists;
  std::vector<int > m_unpack_index_list_lengths;
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


#endif // RAJA_EXAMPLES_HALOEXCHANGE_ITEM_HPP
