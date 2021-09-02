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
#include "SumTransaction.hpp"


Item::Item(double* var,
           Order pack_transaction_order,
           std::vector<int*>& pack_index_lists,
           std::vector<int >& pack_index_list_lengths,
           Order unpack_transaction_order,
           std::vector<int*>& unpack_index_lists,
           std::vector<int >& unpack_index_list_lengths,
           TransactionType transaction_type)
  : m_var(var)
  , m_pack_transaction_order(pack_transaction_order)
  , m_pack_index_lists(pack_index_lists)
  , m_pack_index_list_lengths(pack_index_list_lengths)
  , m_unpack_transaction_order(unpack_transaction_order)
  , m_unpack_index_lists(unpack_index_lists)
  , m_unpack_index_list_lengths(unpack_index_list_lengths)
  , m_transaction_type(transaction_type)
{

}

void Item::populate(Schedule& schedule)
{
  int num_neighbors = m_pack_index_lists.size();

  assert(num_neighbors == m_pack_index_list_lengths.size());
  assert(num_neighbors == m_unpack_index_lists.size());
  assert(num_neighbors == m_unpack_index_list_lengths.size());
  assert(m_transaction_type == TransactionType::copy ||
         m_transaction_type == TransactionType::sum);

  for (int l = 0; l < num_neighbors; ++l) {

    int neighbor_rank = l + 1;

    int* pack_list = m_pack_index_lists[l];
    int  pack_len  = m_pack_index_list_lengths[l];

    FusibleTransaction* pack = nullptr;
    if (m_transaction_type == TransactionType::copy)
    {
      pack = new CopyTransaction(schedule.get_my_rank(),
                                 neighbor_rank,
                                 m_var,
                                 pack_list, pack_len);
    }
    else if (m_transaction_type == TransactionType::sum)
    {
      pack = new SumTransaction(schedule.get_my_rank(),
                                neighbor_rank,
                                m_var,
                                pack_list, pack_len);
    }
    else
    {
      //error
    }

    if (m_pack_transaction_order == Order::unordered && get_loop_pattern_fusible()) {
      schedule.appendTransaction(std::unique_ptr<FusibleTransaction>(pack));
    } else {
      schedule.appendTransaction(std::unique_ptr<Transaction>(pack));
    }


    int* recv_list = m_unpack_index_lists[l];
    int  recv_len  = m_unpack_index_list_lengths[l];

    FusibleTransaction* recv = nullptr;
    if (m_transaction_type == TransactionType::copy)
    {
     recv = new CopyTransaction(neighbor_rank,
                                schedule.get_my_rank(),
                                m_var,
                                recv_list, recv_len);
    }
    else if (m_transaction_type == TransactionType::sum)
    {
      recv = new SumTransaction(neighbor_rank,
                                schedule.get_my_rank(),
                                m_var,
                                recv_list, recv_len);
    }
    else  
    {
      //error  
    }

    if (m_unpack_transaction_order == Order::unordered && get_loop_pattern_fusible()) {
      schedule.appendTransaction(std::unique_ptr<FusibleTransaction>(recv));
    } else {
      schedule.appendTransaction(std::unique_ptr<Transaction>(recv));
    }

  }
}

void Item::populate(GraphSchedule& graphSchedule)
{
  int num_neighbors = m_pack_index_lists.size();

  assert(num_neighbors == m_pack_index_list_lengths.size());
  assert(num_neighbors == m_unpack_index_lists.size());
  assert(num_neighbors == m_unpack_index_list_lengths.size());
  assert(m_transaction_type == TransactionType::copy ||
         m_transaction_type == TransactionType::sum);

  for (int l = 0; l < num_neighbors; ++l) {

    int neighbor_rank = l + 1;

    int* pack_list = m_pack_index_lists[l];
    int  pack_len  = m_pack_index_list_lengths[l];

    FusibleTransaction* pack = nullptr;
    if (m_transaction_type == TransactionType::copy)
    {
      pack = new CopyTransaction(graphSchedule.get_my_rank(),
                                 neighbor_rank,
                                 m_var,
                                 pack_list, pack_len);
    }
    else if (m_transaction_type == TransactionType::sum)
    {
      pack = new SumTransaction(graphSchedule.get_my_rank(),
                                neighbor_rank,
                                m_var,
                                pack_list, pack_len);
    }
    else
    {
      //error
    }

    if (m_pack_transaction_order == Order::unordered && get_loop_pattern_fusible()) {
      graphSchedule.appendTransaction(std::unique_ptr<FusibleTransaction>(pack));
    } else {
      graphSchedule.appendTransaction(std::unique_ptr<Transaction>(pack));
    }


    int* recv_list = m_unpack_index_lists[l];
    int  recv_len  = m_unpack_index_list_lengths[l];

    FusibleTransaction* recv = nullptr;
    if (m_transaction_type == TransactionType::copy)
    {
      recv = new CopyTransaction(neighbor_rank,
                                 graphSchedule.get_my_rank(),
                                 m_var,
                                 recv_list, recv_len);
    }
    else if (m_transaction_type == TransactionType::sum)
    {
      recv = new SumTransaction(neighbor_rank,
                                graphSchedule.get_my_rank(),
                                m_var,
                                recv_list, recv_len);
    }
    else
    {
      //error  
    }

    if (m_unpack_transaction_order == Order::unordered && get_loop_pattern_fusible()) {
      graphSchedule.appendTransaction(std::unique_ptr<FusibleTransaction>(recv));
    } else {
      graphSchedule.appendTransaction(std::unique_ptr<Transaction>(recv));
    }

  }
}
