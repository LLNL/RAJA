//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_TRANSACTION_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_TRANSACTION_HPP

#include <cstddef>

#include "MessageStream.hpp"


struct Transaction
{
  Transaction(int source_processor,
              int destination_processor)
    : m_source_processor(source_processor)
    , m_destination_processor(destination_processor)
  { }

  Transaction(Transaction const&) = delete;
  Transaction& operator=(Transaction const&) = delete;

  virtual ~Transaction() = default;

  virtual size_t computeIncomingMessageSize() = 0;

  virtual size_t computeOutgoingMessageSize() = 0;

  virtual void packStream(MessageStream& stream) = 0;

  virtual void unpackStream(MessageStream& stream) = 0;

  int getSourceProcessor()
  {
    return m_source_processor;
  }

  int getDestinationProcessor()
  {
    return m_destination_processor;
  }

private:
  int m_source_processor;
  int m_destination_processor;
};

#endif // RAJA_EXAMPLES_HALOEXCHANGE_TRANSACTION_HPP
