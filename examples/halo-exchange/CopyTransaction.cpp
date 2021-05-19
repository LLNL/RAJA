//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CopyTransaction.hpp"

#include "halo-exchange.hpp"


size_t CopyTransaction::computeIncomingMessageSize()
{
  return MessageStream::getSizeof<double>(m_len);
}

size_t CopyTransaction::computeOutgoingMessageSize()
{
  return MessageStream::getSizeof<double>(m_len);
}

void CopyTransaction::packStream(MessageStream& stream)
{
  const int     len     = m_len;
  const    int* indices = m_indices;
  const double* var     = m_var;
        double* buf     = stream.getWriteBuffer<double>(len);

  RAJA::forall<raja_forall_sequential_policy>(
      RAJA::TypedRangeSegment<int>(0, len),
      [=] RAJA_HOST_DEVICE (int i) {
    buf[i] = var[indices[i]];
  });
}

void CopyTransaction::unpackStream(MessageStream& stream)
{
  const int     len     = m_len;
  const    int* indices = m_indices;
        double* var     = m_var;
  const double* buf     = stream.getReadBuffer<double>(len);

  RAJA::forall<raja_forall_sequential_policy>(
      RAJA::TypedRangeSegment<int>(0, len),
      [=] RAJA_HOST_DEVICE (int i) {
    var[indices[i]] = buf[i];
  });
}
