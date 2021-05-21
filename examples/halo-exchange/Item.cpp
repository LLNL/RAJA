//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstddef>

#include "loop.hpp"
#include "Item.hpp"
#include "CopyTransaction.hpp"


Item::Item(Order pack_order, Order unpack_order,
     double* var,
     std::vector<int*>& pack_index_lists,
     std::vector<int >& pack_index_list_lengths,
     std::vector<int*>& unpack_index_lists,
     std::vector<int >& unpack_index_list_lengths)
  : m_pack_order(pack_order)
  , m_unpack_order(unpack_order)
  , m_var(var)
  , m_pack_index_lists(pack_index_lists)
  , m_pack_index_list_lengths(pack_index_list_lengths)
  , m_unpack_index_lists(unpack_index_lists)
  , m_unpack_index_list_lengths(unpack_index_list_lengths)
{

}

void Item::populate(Schedule& schedule)
{
  int num_neighbors = m_pack_index_lists.size();

  assert(num_neighbors == m_pack_index_list_lengths.size());
  assert(num_neighbors == m_unpack_index_lists.size());
  assert(num_neighbors == m_unpack_index_list_lengths.size());

  for (int l = 0; l < num_neighbors; ++l) {

    int neighbor_rank = l + 1;

    int* pack_list = m_pack_index_lists[l];
    int  pack_len  = m_pack_index_list_lengths[l];

    CopyTransaction* pack =
        new CopyTransaction(schedule.get_my_rank(),
                            neighbor_rank,
                            m_var,
                            pack_list, pack_len);

    if (m_pack_order == Order::unordered && get_loop_pattern_fusible()) {
      schedule.appendTransaction(std::unique_ptr<FusibleTransaction>(pack));
    } else {
      schedule.appendTransaction(std::unique_ptr<Transaction>(pack));
    }


    int* recv_list = m_unpack_index_lists[l];
    int  recv_len  = m_unpack_index_list_lengths[l];

    CopyTransaction* recv =
        new CopyTransaction(neighbor_rank,
                            schedule.get_my_rank(),
                            m_var,
                            recv_list, recv_len);

    if (m_unpack_order == Order::unordered && get_loop_pattern_fusible()) {
      schedule.appendTransaction(std::unique_ptr<FusibleTransaction>(recv));
    } else {
      schedule.appendTransaction(std::unique_ptr<Transaction>(recv));
    }

  }
}
