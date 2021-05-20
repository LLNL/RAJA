//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_SCHEDULE_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_SCHEDULE_HPP

#include <cstddef>
#include <list>
#include <map>
#include <memory>

#include "Transaction.hpp"
#include "MessageStream.hpp"


struct Schedule
{
  Schedule(int my_rank)
    : m_my_rank(my_rank)
  { }

  Schedule(const Schedule&) = delete;
  Schedule& operator=(const Schedule&) = delete;

  ~Schedule();

  void addTransaction(std::unique_ptr<Transaction>&& transaction);

  void appendTransaction(std::unique_ptr<Transaction>&& transaction);

  void addTransaction(std::unique_ptr<FusibleTransaction>&& transaction);

  void appendTransaction(std::unique_ptr<FusibleTransaction>&& transaction);

  void communicate();

  void beginCommunication();

  void finalizeCommunication();

  void setDeterministicUnpackOrderingFlag(bool flag)
  {
    m_unpack_in_deterministic_order = flag;
  }

private:
  struct TransactionSet
  {
    std::list< std::unique_ptr<Transaction> >        transactions;
    std::list< std::unique_ptr<FusibleTransaction> > fusible_transactions;
  };
  using TransactionSets = std::map<int, TransactionSet>;
  // TransactionSet  m_local_set;
  TransactionSets m_send_sets;
  TransactionSets m_recv_sets;

  // void* m_local_fuser = nullptr;
  void* m_pack_fuser = nullptr;
  void* m_recv_fuser = nullptr;


  struct Buffer
  {
    void* buffer;
    size_t fusible_size;
    size_t size;
  };
  using BufferSet = std::map<int, Buffer>;
  BufferSet m_buffers;

  const int m_my_rank;

  bool m_unpack_in_deterministic_order = true;

  void postReceives();
  void postSends();
  // void performLocalCopies();
  void processCompletedCommunications();
  void deallocateSendBuffers();
};

#endif // RAJA_EXAMPLES_HALOEXCHANGE_SCHEDULE_HPP
