//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_MESSAGESTREAM_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_MESSAGESTREAM_HPP

#include <cassert>
#include <cstddef>


struct MessageStream
{
  enum StreamMode { Read, Write };

  MessageStream() = delete;

  MessageStream(const StreamMode mode, void* buffer, const size_t num_bytes)
    : m_mode(mode), m_buffer(static_cast<char*>(buffer)), m_buffer_size(num_bytes)
  { }

  MessageStream(const MessageStream&) = delete;
  MessageStream& operator=(const MessageStream&) = delete;

  ~MessageStream() = default;

  template<typename DATA_TYPE>
  static size_t getSizeof(size_t num_items)
  {
    return num_items * sizeof(DATA_TYPE);
  }

  void* getBufferStart() const
  {
    return static_cast<void *>(m_buffer);
  }

  size_t getCurrentSize() const
  {
    return m_buffer_index;
  }

  template<typename DATA_TYPE>
  const DATA_TYPE* getReadBuffer(size_t num_entries)
  {
    assert(readMode());
    const size_t num_bytes = getSizeof<DATA_TYPE>(num_entries);
    assert(canAdvance(num_bytes));
    const DATA_TYPE *buffer =
      reinterpret_cast<const DATA_TYPE*>(&m_buffer[getCurrentSize()]);
    m_buffer_index += num_bytes;
    return buffer;
  }

  template<typename DATA_TYPE>
  DATA_TYPE* getWriteBuffer(size_t num_entries)
  {
    assert(writeMode());
    const size_t num_bytes = getSizeof<DATA_TYPE>(num_entries);
    assert(canAdvance(num_bytes));
    DATA_TYPE *buffer =
      reinterpret_cast<DATA_TYPE*>(&m_buffer[getCurrentSize()]);
    m_buffer_index += num_bytes;
    return buffer;
  }

  bool readMode() const
  {
    return m_mode == Read;
  }

  bool writeMode() const
  {
    return m_mode == Write;
  }

  bool canAdvance(size_t num_bytes) const
  {
    return m_buffer_index + num_bytes <= m_buffer_size;
  }

private:
  const StreamMode m_mode;
  char* m_buffer;
  size_t m_buffer_size;
  size_t m_buffer_index = 0;
};

#endif // RAJA_EXAMPLES_HALOEXCHANGE_MESSAGESTREAM_HPP
