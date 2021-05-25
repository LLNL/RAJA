//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstddef>
#include <cassert>

#include "GraphSchedule.hpp"
#include "loop.hpp"


GraphSchedule::~GraphSchedule()
{
  // if (m_local_fuser != nullptr) {
  //   loop_fuser_destroy(m_local_fuser);
  // }
  if (m_recv_fuser != nullptr) {
    loop_fuser_destroy(m_recv_fuser);
  }
  if (m_pack_fuser != nullptr) {
    loop_fuser_destroy(m_pack_fuser);
  }
}

void GraphSchedule::addTransaction(std::unique_ptr<Transaction>&& transaction)
{
  const int src_id = transaction->getSourceProcessor();
  const int dst_id = transaction->getDestinationProcessor();

  if ((m_my_rank == src_id) && (m_my_rank == dst_id)) {
    // m_local_set.transactions.emplace_front(std::move(transaction));
    assert(0);
  } else {
    if (m_my_rank == dst_id) {
      m_recv_sets[src_id].transactions.emplace_front(std::move(transaction));
    } else if (m_my_rank == src_id) {
      m_send_sets[dst_id].transactions.emplace_front(std::move(transaction));
    }
  }
}

void GraphSchedule::appendTransaction(std::unique_ptr<Transaction>&& transaction)
{
  const int src_id = transaction->getSourceProcessor();
  const int dst_id = transaction->getDestinationProcessor();

  if ((m_my_rank == src_id) && (m_my_rank == dst_id)) {
    // m_local_set.transactions.emplace_back(std::move(transaction));
    assert(0);
  } else {
    if (m_my_rank == dst_id) {
      m_recv_sets[src_id].transactions.emplace_back(std::move(transaction));
    } else if (m_my_rank == src_id) {
      m_send_sets[dst_id].transactions.emplace_back(std::move(transaction));
    }
  }
}

void GraphSchedule::addTransaction(std::unique_ptr<FusibleTransaction>&& transaction)
{
  const int src_id = transaction->getSourceProcessor();
  const int dst_id = transaction->getDestinationProcessor();

  if ((m_my_rank == src_id) && (m_my_rank == dst_id)) {
    // if (m_local_fuser == nullptr) {
    //   m_local_fuser = loop_fuser_create();
    // }
    // transaction->set_fuser(m_local_fuser);
    // m_local_set.fusible_transactions.emplace_front(std::move(transaction));
    assert(0);
  } else {
    if (m_my_rank == dst_id) {
      if (m_recv_fuser == nullptr) {
        m_recv_fuser = loop_fuser_create();
      }
      transaction->set_fuser(m_recv_fuser);
      m_recv_sets[src_id].fusible_transactions.emplace_front(std::move(transaction));
    } else if (m_my_rank == src_id) {
      if (m_pack_fuser == nullptr) {
        m_pack_fuser = loop_fuser_create();
      }
      transaction->set_fuser(m_pack_fuser);
      m_send_sets[dst_id].fusible_transactions.emplace_front(std::move(transaction));
    }
  }
}

void GraphSchedule::appendTransaction(std::unique_ptr<FusibleTransaction>&& transaction)
{
  const int src_id = transaction->getSourceProcessor();
  const int dst_id = transaction->getDestinationProcessor();

  if ((m_my_rank == src_id) && (m_my_rank == dst_id)) {
    // if (m_local_fuser == nullptr) {
    //   m_local_fuser = loop_fuser_create();
    // }
    // transaction->set_fuser(m_local_fuser);
    // m_local_set.fusible_transactions.emplace_back(std::move(transaction));
    assert(0);
  } else {
    if (m_my_rank == dst_id) {
      if (m_recv_fuser == nullptr) {
        m_recv_fuser = loop_fuser_create();
      }
      transaction->set_fuser(m_recv_fuser);
      m_recv_sets[src_id].fusible_transactions.emplace_back(std::move(transaction));
    } else if (m_my_rank == src_id) {
      if (m_pack_fuser == nullptr) {
        m_pack_fuser = loop_fuser_create();
      }
      transaction->set_fuser(m_pack_fuser);
      m_send_sets[dst_id].fusible_transactions.emplace_back(std::move(transaction));
    }
  }
}

void GraphSchedule::communicate()
{
  beginCommunication();
  finalizeCommunication();
}

void GraphSchedule::beginCommunication()
{
  postReceives();
  postSends();
}

void GraphSchedule::finalizeCommunication()
{
  // performLocalCopies();
  processCompletedCommunications();
}

void GraphSchedule::postReceives()
{
  TransactionSets::iterator mi = m_recv_sets.lower_bound(m_my_rank);

  for (size_t counter = 0;
       counter < m_recv_sets.size();
       ++counter) {

    if (mi == m_recv_sets.begin()) {
      mi = m_recv_sets.end();
    }
    --mi;

    size_t byte_count = 0;
    for (std::unique_ptr<FusibleTransaction>& recv : mi->second.fusible_transactions) {
      byte_count += recv->computeIncomingMessageSize();
    }
    const size_t fusible_byte_count = byte_count;
    for (std::unique_ptr<Transaction>& recv : mi->second.transactions) {
      byte_count += recv->computeIncomingMessageSize();
    }

    void* buffer = loop_allocate_buffer(byte_count);
    m_recv_buffers[mi->first] = Buffer{buffer, fusible_byte_count, byte_count};

    // irecv one
  }
}

void
GraphSchedule::postSends()
{
  for (auto& send_data : m_send_sets) {

    size_t byte_count = 0;
    for (std::unique_ptr<FusibleTransaction>& pack : send_data.second.fusible_transactions) {
      byte_count += pack->computeIncomingMessageSize();
    }
    const size_t fusible_byte_count = byte_count;
    for (std::unique_ptr<Transaction>& pack : send_data.second.transactions) {
      byte_count += pack->computeIncomingMessageSize();
    }

    // because communication is fake send and recv buffers are the same
    Buffer& buffer = m_recv_buffers[send_data.first];
    assert(buffer.size == byte_count);
    m_pack_buffers[send_data.first] = Buffer{buffer.buffer, fusible_byte_count, byte_count};

    // Pack outgoing data into a message.
    MessageStream outgoing_stream(
        MessageStream::Write,
        buffer.buffer,
        fusible_byte_count);

    for (std::unique_ptr<FusibleTransaction>& pack : send_data.second.fusible_transactions) {
      pack->packStream(outgoing_stream);
    }

  }

  if (m_pack_fuser != nullptr) {
    loop_fuser_run(m_pack_fuser);
    loop_synchronize();
    loop_fuser_clear(m_pack_fuser);
  }

  TransactionSets::iterator mi = m_send_sets.upper_bound(m_my_rank);

  for (size_t counter = 0;
       counter < m_send_sets.size();
       ++counter, ++mi) {

    if (mi == m_send_sets.end()) {
      mi = m_send_sets.begin();
    }

    Buffer& buffer = m_pack_buffers[mi->first];

    // Pack outgoing data into a message.
    MessageStream outgoing_stream(
        MessageStream::Write,
        static_cast<char*>(buffer.buffer) + buffer.fusible_size,
        buffer.size - buffer.fusible_size);

    for (std::unique_ptr<Transaction>& pack : mi->second.transactions) {
      pack->packStream(outgoing_stream);
    }

    if (!mi->second.transactions.empty()) {
      loop_synchronize();
    }

    // isend one
  }
}

// void GraphSchedule::performLocalCopies()
// {
//   for (std::unique_ptr<Transaction>& local : m_local_set.fusible_transactions) {
//     local->copyLocalData();
//   }
//   for (std::unique_ptr<Transaction>& local : m_local_set.transactions) {
//     local->copyLocalData();
//   }

//   if (m_local_fuser != nullptr) {
//     loop_fuser_run(m_local_fuser);
//     loop_synchronize();
//     loop_fuser_clear(m_local_fuser);
//   }
// }

void GraphSchedule::processCompletedCommunications()
{
  if (m_unpack_in_deterministic_order) {

    // wait recv all

    for (auto& recv_data : m_recv_sets) {

      Buffer& buffer = m_recv_buffers[recv_data.first];

      MessageStream incoming_stream(
          MessageStream::Read,
          buffer.buffer,
          buffer.size);

      for (std::unique_ptr<FusibleTransaction>& recv : recv_data.second.fusible_transactions) {
        recv->unpackStream(incoming_stream);
      }
      for (std::unique_ptr<Transaction>& recv : recv_data.second.transactions) {
        recv->unpackStream(incoming_stream);
      }

    }

  } else {

    for (size_t counter = m_recv_sets.size();
         counter > 0u;
         --counter) {

      // wait recv one

      // TODO: Make general this assumes senders in [1,m_recv_sets.size()]
      const int sender = counter;
      assert(m_my_rank == 0);

      Buffer& buffer = m_recv_buffers[sender];

      MessageStream incoming_stream(
          MessageStream::Read,
          buffer.buffer,
          buffer.size);

      for (std::unique_ptr<FusibleTransaction>& recv : m_recv_sets[sender].fusible_transactions) {
        recv->unpackStream(incoming_stream);
      }
      for (std::unique_ptr<Transaction>& recv : m_recv_sets[sender].transactions) {
        recv->unpackStream(incoming_stream);
      }

    }

  }

  if (m_recv_fuser != nullptr) {
    loop_fuser_run(m_recv_fuser);
  }

  loop_synchronize();

  if (m_recv_fuser != nullptr) {
    loop_fuser_clear(m_recv_fuser);
  }

  // wait send all

  for (auto& bufferData : m_recv_buffers) {

    Buffer& buffer = bufferData.second;

    loop_deallocate_buffer(buffer.buffer);
    buffer.buffer = nullptr;
    buffer.fusible_size = 0;
    buffer.size = 0;
  }

  for (auto& bufferData : m_pack_buffers) {

    Buffer& buffer = bufferData.second;

    buffer.buffer = nullptr;
    buffer.fusible_size = 0;
    buffer.size = 0;
  }
}
