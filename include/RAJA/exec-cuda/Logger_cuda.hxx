/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   
 *
 ******************************************************************************
 */

#ifndef RAJA_CUDA_Logger_HXX
#define RAJA_CUDA_Logger_HXX

#include "RAJA/config.hxx"

#ifdef RAJA_ENABLE_CUDA

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/Logger.hxx"

#include <type_traits>

#include <stdio.h>
#include <stdint.h>

namespace RAJA
{

namespace Internal
{

  enum struct print_types: char {
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    flt32,
    flt64,
    ptr,
    char_ptr,
    char_arr
  };

class CudaLogManager {
public:
  static const int buffer_size = 1024*1024;
  static const int error_buffer_size = 4*1024;

  static inline CudaLogManager* getInstance()
  {
    static CudaLogManager* me = new (allocateInstance()) CudaLogManager();
    return me;
  }

  static void s_check_logs()
  {
    if (s_instance_buffer != nullptr) {
      getInstance()->check_logs();
    }
  }

  void check_logs()
  {
    if (m_flag) {
      // handle logs
      handle_in_order();
    }
  }

  void handle_in_order()
  {
    // ensure all logs visible
    cudaDeviceSynchronize();

    loggingID_type error_id = std::numeric_limits<loggingID_type>::max();
    loggingID_type log_id = std::numeric_limits<loggingID_type>::max();

    char* error_pos = m_error_begin;
    char* error_next = m_error_pos;
    char* error_end = m_error_pos;
    read_preamble(error_pos, error_next, error_end, error_id);

    char* log_pos = m_log_begin;
    char* log_next = m_log_pos;
    char* log_end = m_log_pos;
    read_preamble(log_pos, log_next, log_end, log_id);

    // handle logs in order by id
    while( error_id != std::numeric_limits<loggingID_type>::max()
        || log_id   != std::numeric_limits<loggingID_type>::max() ) {

      if (error_id < log_id) {
        handle_error(error_pos, error_next);
        read_preamble(error_pos, error_next, error_end, error_id);
      } else {
        handle_log(log_pos, log_next);
        read_preamble(log_pos, log_next, log_end, log_id);
      }

    }

    // reset buffers and flags
    m_error_pos = m_error_begin;
    memset(m_error_begin, 0, m_error_end - m_error_begin);

    m_log_pos = m_log_begin;
    memset(m_log_begin, 0, m_log_end - m_log_begin);

    m_flag = false;
  }

  template < typename T_fmt, typename... T >
  RAJA_HOST_DEVICE
  void
  error_impl(RAJA::logging_function_type func, int udata, T_fmt fmt, T... args)
  {
#ifdef __CUDA_ARCH__
    while ( atomicCAS(&md_data->mutex, 0, 1) != 0 );
    loggingID_type num = atomicAdd(&md_data->count, 1);
    char* buf_pos = m_error_pos;
    char* buf_end = m_error_end;

    buf_pos = write_log(buf_pos, buf_end, num, func, udata, fmt, args...);

    m_error_pos = buf_pos;
    __threadfench_system();
    m_flag = true;
    atomicExch(&md_data->mutex, 0);
#else
    cudaDeviceSynchronize();
    check_logs();
    ResourceHandler::getInstance().error_cleanup(RAJA::error::user);
    func(udata, fmt);
    exit(1);
#endif
  }

  template < typename T_fmt, typename... T >
  RAJA_HOST_DEVICE
  void
  log_impl(RAJA::logging_function_type func, int udata, T_fmt fmt, T... args)
  {
#ifdef __CUDA_ARCH__
    while ( atomicCAS(&md_data->mutex, 0, 1) != 0 );
    loggingID_type num = atomicAdd(&md_data->count, 1);
    char* buf_pos = m_log_pos;
    char* buf_end = m_log_end;

    buf_pos = write_log(buf_pos, buf_end, num, func, udata, fmt, args...);

    m_log_pos = buf_pos;
    __threadfench_system();
    m_flag = true;
    atomicExch(&md_data->mutex, 0);
#else
    check_logs();
    func(udata, fmt);
#endif
  }

private:
  struct CudaLogManagerDeviceData
  {
    int mutex = 0;
    loggingID_type count = 0;
  };

  bool  m_flag = false;
  char* m_error_begin = nullptr;
  char* m_error_pos = nullptr;
  char* m_error_end = nullptr;
  char* m_log_begin = nullptr;
  char* m_log_pos = nullptr;
  char* m_log_end = nullptr;
  CudaLogManagerDeviceData* md_data = nullptr;

  static char* s_instance_buffer;

  static void* allocateInstance()
  {
    if (s_instance_buffer == nullptr) {
      cudaHostAlloc(&s_instance_buffer, buffer_size, cudaHostAllocDefault);
      memset(s_instance_buffer, 0, buffer_size);
    }
    return s_instance_buffer;
  }

  static void deallocateInstance()
  {
    if (s_instance_buffer != nullptr) {
      getInstance()->~CudaLogManager();
      cudaFreeHost(s_instance_buffer);
    }
  }

  CudaLogManager()
    : m_flag(false),
      m_error_begin(s_instance_buffer + sizeof(CudaLogManager)),
      m_error_pos(s_instance_buffer + sizeof(CudaLogManager)),
      m_error_end(s_instance_buffer + sizeof(CudaLogManager) + error_buffer_size),
      m_log_begin(s_instance_buffer + sizeof(CudaLogManager) + error_buffer_size),
      m_log_pos(s_instance_buffer + sizeof(CudaLogManager) + error_buffer_size),
      m_log_end(s_instance_buffer + buffer_size),
      md_data(nullptr)
  {
    cudaMalloc((void**)&md_data, sizeof(CudaLogManagerDeviceData));
    cudaMemset(md_data, 0, sizeof(CudaLogManagerDeviceData));
  }

  ~CudaLogManager()
  {
    cudaFree(md_data);
  }

  template < typename T_fmt, typename... T >
  RAJA_HOST_DEVICE
  static char* 
  write_log(char* buf_pos, char* buf_end, loggingID_type id, RAJA::logging_function_type func, int udata, T_fmt fmt, T... args)
  {
    char* init_buf_pos = buf_pos;
    buf_pos += sizeof(char*);
    buf_pos = write_value(buf_pos, buf_end, id);
    buf_pos = write_value(buf_pos, buf_end, func);
    buf_pos = write_value(buf_pos, buf_end, udata);
    buf_pos = write_types_values(buf_pos, buf_end, fmt, args...);
    write_value(init_buf_pos, buf_end, buf_pos);

    return buf_pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static char*
  write_type(char* pos, char* end, T const&)
  {
    static_assert(!std::is_same<T, void>::value || std::is_same<T, void>::value, "Raja error: unknown type in log");
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 1, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::int8);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 2, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::int16);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 4, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::int32);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 8, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::int64);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 1, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::uint8);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 2, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::uint16);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 4, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::uint32);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 8, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::uint64);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_floating_point< T >::value && sizeof(T) == 4, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::flt32);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static typename std::enable_if< std::is_floating_point< T >::value && sizeof(T) == 8, char*>::type
  write_type(char* pos, char* end, T const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::flt64);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static char*
  write_type(char* pos, char* end, T* const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::ptr);
    return pos;
  }

  RAJA_HOST_DEVICE
  static char*
  write_type(char* pos, char* end, const char* const&)
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::char_ptr);
    return pos;
  }

  template < int N >
  RAJA_HOST_DEVICE
  static char*
  write_type(char* pos, char* end, const char (&)[N])
  {
    if (pos < end) *pos++ = static_cast<char>(RAJA::Internal::print_types::char_arr);
    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static char*
  write_value(char* pos, char* end, T arg)
  {
    union Tchar_union {
      T t;
      char ca[sizeof(T)];
    };

    Tchar_union u;
    u.t = arg;

    for (int i = 0; i < sizeof(T); i++) {
      if (pos >= end) break;
      *pos++ = u.ca[i];
    }

    return pos;
  }

  template < int N >
  RAJA_HOST_DEVICE
  static char*
  write_value(char* pos, char* end, const char (& arg)[N])
  {
    for ( int i = 0; i < N; ++i ) {
      if (pos >= end) break;
      if ('\0' == (*pos++ = arg[i])) break;
    }

    return pos;
  }

  template < typename T >
  RAJA_HOST_DEVICE
  static char*
  write_types_values(char* pos, char* end, T arg)
  {
    pos = write_type(pos, end, arg);
    pos = write_value(pos, end, arg);
    return pos;
  }

  template < typename T, typename... Ts >
  RAJA_HOST_DEVICE
  static char*
  write_types_values(char* pos, char* end, T arg, Ts... args)
  {
    pos = write_type(pos, end, arg);
    pos = write_value(pos, end, arg);
    pos = write_values(pos, end, args...);
    return pos;
  }

  // functions that read from the buffer

  static bool
  read_preamble(char*& buf_pos, char*& buf_next, char*const& buf_end, loggingID_type& id)
  {
    char* tmp_next;
    bool err = read_value<char*>(buf_pos, buf_next, tmp_next);
    if ( !err ) {
      buf_next = tmp_next;
      err = read_value<loggingID_type>(buf_pos, buf_next, id);
    }
    if ( err ) {
      buf_pos = buf_end;
      buf_next = buf_end;
      id = std::numeric_limits<loggingID_type>::max();
    }
    return err;
  }

  static bool
  handle_error(char*& buf_pos, char*const& buf_end)
  {
    ResourceHandler::getInstance().error_cleanup(RAJA::error::user);
    bool err = handle_log(buf_pos, buf_end);
    exit(1);
    return err;
  }

  static bool
  handle_log(char*& buf_pos, char*const& buf_end)
  {
    bool err = false;
    RAJA::logging_function_type func;
    int udata;
    const char* fmt;

    // read the logging function, udata, and fmt
    err = read_value<RAJA::logging_function_type>(buf_pos, buf_end, func) || err;
    err = read_value<int>(buf_pos, buf_end, udata) || err;
    err = read_type_value(buf_pos, buf_end, fmt) || err;

    // read the other arguments and print to another buffer.
    // err = write_types_values(buf_pos, buf_end, args...) || err;

    if ( !err ) {
      func(udata, fmt);
    } else {
      fprintf(stderr, "RAJA Logging error: Logging function could not be called.");
      buf_pos = buf_end;
    }

    if (buf_pos != buf_end) {
      fprintf(stderr, "RAJA Logging warning: Unused arguments detected.");
      buf_pos = buf_end;
    }

    return err;
  }

  template < typename T_read, typename T >
  static typename std::enable_if< std::is_assignable<T, T_read>::value, bool >::type
  read_value(char*& pos, char*const& end, T& arg)
  {
    bool err = false;
    union Tchar_union {
      T_read t;
      char ca[sizeof(T_read)];
    };

    Tchar_union u;

    for (int i = 0; i < sizeof(T_read); i++) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.");
        u.t = static_cast<T_read>(0);
        err = true;
        break;
      }
      u.ca[i] = *pos++;
    }

    arg = static_cast<T>(u.t);

    return err;
  }

  template < typename T_read, typename T >
  static typename std::enable_if< !std::is_assignable<T, T_read>::value, bool >::type
  read_value(char*& pos, char*const& end, T& arg)
  {
    bool err = false;
    for (int i = 0; i < sizeof(T_read); i++) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.");
        err = true;
        break;
      }
      pos++;
    }

    fprintf(stderr, "RAJA logger warning: Incompatible types detected");
    arg = static_cast<T>(0);

    return err;
  }

  template < typename T >
  static typename std::enable_if< std::is_same< typename std::decay<T>::type, char*>::value, bool >::type
  read_value_char_arr(char*& pos, char*const& end, T& arg)
  {
    bool err = false;
    arg = static_cast<T>(pos);
    while (!err) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.");
        arg = static_cast<T>(0);
        err = true;
        break;
      }
      if ( *pos++ == '\0') break;
    }
    return err;
  }

  template < typename T >
  static typename std::enable_if< !std::is_same< typename std::decay<T>::type, char*>::value, bool >::type
  read_value_char_arr(char*& pos, char*const& end, T& arg)
  {
    fprintf(stderr, "RAJA logger warning: Incompatible types detected");
    bool err = false;
    arg = static_cast<T>(0);
    while (!err) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.");
        err = true;
        break;
      }
      if ( *pos++ == '\0') break;
    }
    return err;
  }

  template < typename T >
  static bool
  read_type_value(char*& pos, char*const& end, T& arg)
  {
    bool err = false;
    if (pos >= end) {
      fprintf(stderr, "RAJA logger warning: Incomplete log detected.");
      err = true;
      return err;
    }

    using print_types = RAJA::Internal::print_types;
    switch( static_cast<print_types>(*pos++) ) {
      case print_types::int8 : {
        err = read_value<int8_t>(pos, end, arg);
        break;
      }
      case print_types::int16 : {
        err = read_value<int16_t>(pos, end, arg);
        break;
      }
      case print_types::int32 : {
        err = read_value<int32_t>(pos, end, arg);
        break;
      }
      case print_types::int64 : {
        err = read_value<int64_t>(pos, end, arg);
        break;
      }
      case print_types::uint8 : {
        err = read_value<uint8_t>(pos, end, arg);
        break;
      }
      case print_types::uint16 : {
        err = read_value<uint16_t>(pos, end, arg);
        break;
      }
      case print_types::uint32 : {
        err = read_value<uint32_t>(pos, end, arg);
        break;
      }
      case print_types::uint64 : {
        err = read_value<uint64_t>(pos, end, arg);
        break;
      }
      case print_types::flt32 : {
        err = read_value<float>(pos, end, arg);
        break;
      }
      case print_types::flt64 : {
        err = read_value<double>(pos, end, arg);
        break;
      }
      case print_types::ptr : {
        err = read_value<void*>(pos, end, arg);
        break;
      }
      case print_types::char_ptr : {
        err = read_value<char*>(pos, end, arg);
        break;
      }
      case print_types::char_arr : {
        err = read_value_char_arr(pos, end, arg);
        break;
      }
      default : {
        fprintf(stderr, "RAJA logger warning: Unknown type encountered");
        err = true;
        break;
      }
    }

    return err;
  }

};

} // closing brace for Internal namespace

template < >
class Logger< RAJA::cuda_logger > {
public:
  using func_type = RAJA::logging_function_type;
  template < typename T, size_t N >
  using array_type = T[N];
  template < typename T >
  using ptr_type = T*;

  explicit Logger(func_type f = RAJA::basic_logger)
    : m_func(f),
      m_logman(Internal::CudaLogManager::getInstance())
  {

  }

  template < size_t N, typename... Ts >
  RAJA_HOST_DEVICE
  void error(int udata, array_type<const char, N>const& fmt, Ts const&... args) const
  {
    m_logman->error_impl(m_func, udata, fmt, args...);
  }

  template < size_t N, typename... Ts >
  RAJA_HOST_DEVICE
  void log(int udata, array_type<const char, N>const& fmt, Ts const&... args) const
  {
    m_logman->log_impl(m_func, udata, fmt, args...);
  }

  template < typename... Ts >
  RAJA_HOST_DEVICE
  void error(int udata, ptr_type<const char>const& fmt, Ts const&... args) const
  {
    m_logman->error_impl(m_func, udata, fmt, args...);
  }

  template < typename... Ts >
  RAJA_HOST_DEVICE
  void log(int udata, ptr_type<const char>const& fmt, Ts const&... args) const
  {
    m_logman->log_impl(m_func, udata, fmt, args...);
  }

private:
  const func_type m_func;
  volatile Internal::CudaLogManager* const m_logman;
};

}  // closing brace for RAJA namespace

#endif

#endif  // closing endif for header file include guard
