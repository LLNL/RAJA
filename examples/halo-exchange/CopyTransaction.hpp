//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_COPYTRANSACTION_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_COPYTRANSACTION_HPP

#include <cstddef>

#include "Transaction.hpp"
#include "MessageStream.hpp"


struct CopyTransaction : Transaction
{
  CopyTransaction(int source_processor,
                  int destination_processor,
                  double* var,
                  const int* indices, int len)
    : Transaction(source_processor, destination_processor)
    , m_var(var)
    , m_indices(indices),     m_len(len)
  { }

  CopyTransaction(CopyTransaction const&) = delete;
  CopyTransaction& operator=(CopyTransaction const&) = delete;

  ~CopyTransaction() override = default;

  size_t computeIncomingMessageSize() override;

  size_t computeOutgoingMessageSize() override;

  void packStream(MessageStream& stream) override;

  void unpackStream(MessageStream& stream) override;

private:
  double* m_var;
  const int* m_indices;
  int m_len;
};

#endif // RAJA_EXAMPLES_HALOEXCHANGE_COPYTRANSACTION_HPP
