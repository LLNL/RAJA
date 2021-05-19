//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstddef>
#include <cassert>

#include "Schedule.hpp"
#include "../memoryManager.hpp"


void Schedule::addTransaction(std::unique_ptr<Transaction>&& transaction)
{
  const int src_id = transaction->getSourceProcessor();
  const int dst_id = transaction->getDestinationProcessor();

  if ((m_my_rank == src_id) && (m_my_rank == dst_id)) {
    // m_local_set.emplace_front(std::move(transaction));
    assert(0);
  } else {
    if (m_my_rank == dst_id) {
      m_recv_sets[src_id].emplace_front(std::move(transaction));
    } else if (m_my_rank == src_id) {
      m_send_sets[dst_id].emplace_front(std::move(transaction));
    }
  }
}

void Schedule::appendTransaction(std::unique_ptr<Transaction>&& transaction)
{
  const int src_id = transaction->getSourceProcessor();
  const int dst_id = transaction->getDestinationProcessor();

  if ((m_my_rank == src_id) && (m_my_rank == dst_id)) {
    // m_local_set.emplace_back(std::move(transaction));
    assert(0);
  } else {
    if (m_my_rank == dst_id) {
      m_recv_sets[src_id].emplace_back(std::move(transaction));
    } else if (m_my_rank == src_id) {
      m_send_sets[dst_id].emplace_back(std::move(transaction));
    }
  }
}

void Schedule::communicate()
{
  beginCommunication();
  finalizeCommunication();
}

void Schedule::beginCommunication()
{
  postReceives();
  postSends();
}

void Schedule::finalizeCommunication()
{
  // performLocalCopies();

  // parallel_synchronize();

  processCompletedCommunications();
}

void Schedule::postReceives()
{
  int rank = 0;

  TransactionSets::iterator mi = m_recv_sets.lower_bound(rank);

  for (size_t counter = 0;
       counter < m_recv_sets.size();
       ++counter) {

    if (mi == m_recv_sets.begin()) {
      mi = m_recv_sets.end();
    }
    --mi;

    size_t byte_count = 0;
    for (std::unique_ptr<Transaction>& recv : mi->second) {
      byte_count += recv->computeIncomingMessageSize();
    }

    void* buffer = memoryManager::allocate<char>(byte_count);
    m_buffers[mi->first] = Buffer{buffer, byte_count};
  }
}

void
Schedule::postSends()
{
  int rank = 0;

  TransactionSets::iterator mi = m_send_sets.upper_bound(rank);

  for (size_t counter = 0;
       counter < m_send_sets.size();
       ++counter, ++mi) {

    if (mi == m_send_sets.end()) {
      mi = m_send_sets.begin();
    }

    size_t byte_count = 0;
    for (std::unique_ptr<Transaction>& pack : mi->second) {
      byte_count += pack->computeOutgoingMessageSize();
    }

    Buffer& buffer = m_buffers[mi->first];

    assert(buffer.size == byte_count);

    // Pack outgoing data into a message.
    MessageStream outgoing_stream(
        MessageStream::Write,
        buffer.buffer,
        byte_count);

    for (std::unique_ptr<Transaction>& pack : mi->second) {
      pack->packStream(outgoing_stream);
    }

    // parallel_synchronize();
  }
}

// void Schedule::performLocalCopies()
// {
//   for (std::unique_ptr<Transaction>& local : m_local_set) {
//     local->copyLocalData();
//   }
// }

void Schedule::processCompletedCommunications()
{
  if (m_unpack_in_deterministic_order) {

    for (auto& recv_data : m_recv_sets) {

      Buffer& buffer = m_buffers[recv_data.first];

      MessageStream incoming_stream(
          MessageStream::Read,
          buffer.buffer,
          buffer.size);

      for (std::unique_ptr<Transaction>& recv : recv_data.second) {
        recv->unpackStream(incoming_stream);
      }

      // parallel_synchronize();

      memoryManager::deallocate(buffer.buffer);
      buffer.buffer = nullptr;
      buffer.size = 0;

    }

  } else {

    for (size_t counter = m_recv_sets.size();
         counter > 0u;
         --counter) {

      // TODO: Make general this assumes senders in [1,m_recv_sets.size()]
      const int sender = counter;

      Buffer& buffer = m_buffers[sender];

      MessageStream incoming_stream(
          MessageStream::Read,
          buffer.buffer,
          buffer.size);

      for (std::unique_ptr<Transaction>& recv : m_recv_sets[sender]) {
        recv->unpackStream(incoming_stream);
      }

      // parallel_synchronize();

      memoryManager::deallocate(buffer.buffer);
      buffer.buffer = nullptr;
      buffer.size = 0;

    }

  }
}
