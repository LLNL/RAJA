/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Cuda implementation of RAJA::Logger
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
#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include <type_traits>
#include <vector>

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

namespace RAJA
{

namespace Internal
{

/*!
 * \brief  Type used to read and write logs.
 */
using rw_type = int;

/*!
 * \brief  Class that determines the extent of a rw_type[] needed to cover the
 *         template types.
 *
 * Note:   Each type is padded individually.
 */
template < typename... Ts >
struct rwtype_sizeof_Ts;
///
template < >
struct rwtype_sizeof_Ts < >
{
  static const size_t value = 0;
};
///
template < typename T0, typename... Ts >
struct rwtype_sizeof_Ts < T0, Ts... >
{
  static const size_t value = (sizeof(T0) + sizeof(rw_type) - 1) / sizeof(rw_type)
                            + rwtype_sizeof_Ts<Ts...>::value;
};

/*!
 ******************************************************************************
 *
 * \brief  Cuda Log Manager class.
 *
 * Note:   Restrictions on cuda log formatting:
 *          - %s  arguments are copied by value, so char* args must point to
 *                 memory accessible on the host when the log handler is called.
 *          - \0  embedded null characters in char arrays and string literals
 *                 are not supported.
 *          - %*d use of * (runtime precision specification) is not currently 
 *                 supported.
 *          - %n  is not supported.
 *
 ******************************************************************************
 */
class CudaLogManager {
public:

  /*!
   * \brief  size of the entire intance buffer.
   */
  static const int buffer_size = 1024*1024;

  /*!
   * \brief  size of the intance buffer dedicated to errors.
   */
  static const int error_buffer_size = 4*1024;

  /*!
   * \brief  type used for offsets into log buffer.
   */
  using logpos_type = int;
  

  /*!
   * \brief  static function that deallocates the buffer where the instance of
   *         CudaLogManager and the error and log buffers are stored.
   */
  static void* allocateInstance()
  {
    if (s_instance_buffer == nullptr) {
      cudaHostAlloc(&s_instance_buffer, buffer_size, cudaHostAllocDefault);
      memset(s_instance_buffer, 0, buffer_size);
    }
    return s_instance_buffer;
  }

  /*!
   * \brief  static function that allocates the buffer where the instance of
   *         CudaLogManager and the error and log buffers are stored.
   */
  static void deallocateInstance()
  {
    if (s_instance_buffer != nullptr) {
      getInstance()->~CudaLogManager();
      cudaFreeHost(s_instance_buffer);
    }
  }

  /*!
   * \brief  static function that gets the single nstance of CudaLogManager.
   */
  static inline CudaLogManager* getInstance()
  {
    static CudaLogManager* me = new (allocateInstance()) CudaLogManager();
    return me;
  }

  /*!
   * \brief  static function that checks if logs are waiting on the device, 
   *         and handles them if logs are waiting.
   */
  static void s_check_logs()
  {
    if (s_instance_buffer != nullptr) {
      getInstance()->check_logs();
    }
  }

  /*!
   * \brief  Checks if logs are waiting on the device, and handles them if 
   *         logs are waiting.
   */
  void check_logs() volatile
  {
    // fprintf(stderr, "RAJA logger: s_instance_buffer = %p.\n", s_instance_buffer);
    if (m_flag) {
      // fprintf(stderr, "RAJA logger: found log in queue.\n");
      // handle logs
      handle_in_order();
    }
  }

  /*!
   * \brief  Implementation of error logging on the device.
   *
   * Note:   Only one error can be written on the device.
   */
  template < typename T_fmt, typename... Ts >
  RAJA_HOST_DEVICE
  void
  error_impl(kernelnum_type num, logging_function_type func, udata_type udata, T_fmt const& fmt, Ts const&... args) volatile
  {
#ifdef __CUDA_ARCH__
    if ( atomicCAS(&md_data->mutex, 0, 1) == 0 ) { // lock error buffer
      logpos_type msg_size = LogWriter::rwtype_sizeof_msg(num, func, udata, fmt, args...);

      rw_type* buf_pos = m_error_pos;
      rw_type* buf_end = m_error_end;

      if (buf_pos + msg_size <= buf_end) {
        buf_end = buf_pos + msg_size;
        LogWriter writer(buf_pos, buf_end);
        writer.write(num, func, udata, fmt, args...);
      } else {
        printf("RAJA logger error: Writing error on device failed.\n");
        msg_size = LogWriter::rwtype_sizeof_msg(num, func, udata, "RAJA logger error: Writing error on device failed.");
        buf_end = m_error_end;
        if (buf_pos + msg_size <= buf_end) {
          buf_end = buf_pos + msg_size;
          LogWriter failed_writer(buf_pos, buf_end);
          failed_writer.write(num, func, udata, "RAJA logger error: Writing error on device failed.");
        }
      }

      m_error_pos = buf_end;
      __threadfence_system();
      m_flag = true;
    }
#else
    cudaDeviceSynchronize();
    check_logs();
    ResourceHandler::getInstance().error_cleanup(RAJA::error::user);
    int len = snprintf(nullptr, 0, fmt, args...);
    if (len >= 0) {
      char* msg = new char[len+1];
      snprintf(msg, len+1, fmt, args...);
      func(udata, msg);
      delete[] msg;
    } else {
      fprintf(stderr, "RAJA logger error: could not format message");
    }
    if (s_exit_enabled) {
      exit(1);
    }
#endif
  }

  /*!
   * \brief  Implementation of logging on the device.
   *
   * Note:   logs are written in parallel.
   */
  template < typename T_fmt, typename... Ts >
  RAJA_HOST_DEVICE
  void
  log_impl(kernelnum_type num, logging_function_type func, udata_type udata, T_fmt const& fmt, Ts const&... args) volatile
  {
#ifdef __CUDA_ARCH__
    int warpIdx = ((threadIdx.z * (blockDim.x * blockDim.y))
                + (threadIdx.y * blockDim.x) + threadIdx.x) % WARP_SIZE;
    int warpIdxmask = 1 << warpIdx;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int num_lanes = __popc(mask);
    int warp_log_offset = __popc(mask & (warpIdxmask - 1));
    // msg_size is the same for all threads in the warp
    // msg_size only depends on the types of the arguments
    logpos_type msg_size = LogWriter::rwtype_sizeof_msg(num, func, udata, fmt, args...);

    rw_type* buf_pos = m_log_begin;
    rw_type* buf_end = m_log_end;
    logpos_type buf_length = buf_end - buf_pos;

    logpos_type pos = 0;
    if (warpIdx == first) {
      pos = _atomicAddUpToMax(&md_data->log_pos, num_lanes * msg_size, buf_length);
    }
    pos = RAJA::HIDDEN::shfl(pos, first);

    buf_pos += pos + msg_size * warp_log_offset;
    if ( buf_pos + msg_size <= buf_end ) {
      buf_end = buf_pos + msg_size;

      LogWriter writer(buf_pos, buf_end);
      writer.write(num, func, udata, fmt, args...);
    } else {
      // printf("RAJA logger error: Writing log on device failed.\n");
      // msg_size = LogWriter::rwtype_sizeof_msg(num, func, udata, "RAJA logger error: Writing log on device failed.");
      // LogWriter failed_writer(buf_pos, buf_end);
      // failed_writer.write(num, func, udata, "RAJA logger error: Writing log on device failed.");
    }
    __threadfence_system();
    if (warpIdx == first) {
      m_flag = true;
    }
#else
    int len = snprintf(nullptr, 0, fmt, args...);
    if (len >= 0) {
      char* msg = new char[len+1];
      snprintf(msg, len+1, fmt, args...);
      func(udata, msg);
      delete[] msg;
    } else {
      fprintf(stderr, "RAJA logger error: could not format message");
    }
#endif
  }

private:

  /*!
   * \brief  enum containing types that are valid formatting arguments.
   */
  enum struct print_types: char {
    invalid = 0,
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
    fltL,
    ptr,
    char_ptr,
    char_arr
  };

  /*!
   * \brief  Device data, contatins atomically updateable mutex for errors and 
   *         position for logs.
   */
  struct CudaLogManagerDeviceData
  {
    int mutex = 0;
    logpos_type log_pos = 0;
  };

  bool  m_flag = false;
  rw_type* m_error_begin = nullptr;
  rw_type* m_error_pos = nullptr;
  rw_type* m_error_end = nullptr;
  rw_type* m_log_begin = nullptr;
  // rw_type* m_log_pos = nullptr;
  rw_type* m_log_end = nullptr;
  CudaLogManagerDeviceData* md_data = nullptr;

  /*!
   * \brief  static variable storing the pointer to the intance buffer.
   */
  static char* s_instance_buffer;

  /*!
   * \brief  CudaLogManager constructor, initializes the nmember pointers to
   *         point to approptiate places in the larger buffer, allocates
   *         device memory for device variables.
   */
  CudaLogManager()
    : m_flag(false),
      m_error_begin(reinterpret_cast<rw_type*>(s_instance_buffer + sizeof(CudaLogManager))),
      m_error_pos(reinterpret_cast<rw_type*>(s_instance_buffer + sizeof(CudaLogManager))),
      m_error_end(reinterpret_cast<rw_type*>(s_instance_buffer + sizeof(CudaLogManager) + error_buffer_size)),
      m_log_begin(reinterpret_cast<rw_type*>(s_instance_buffer + sizeof(CudaLogManager) + error_buffer_size)),
      // m_log_pos(reinterpret_cast<rw_type*>(s_instance_buffer + sizeof(CudaLogManager) + error_buffer_size)),
      m_log_end(reinterpret_cast<rw_type*>(s_instance_buffer + buffer_size)),
      md_data(nullptr)
  {
    static_assert(sizeof(CudaLogManager) % sizeof(rw_type) == 0,
                "RAJA logger error: rw_type invalid due to alignment");
    static_assert(error_buffer_size % sizeof(rw_type) == 0,
                "RAJA logger error: rw_type invalid due to alignment");
    cudaMalloc((void**)&md_data, sizeof(CudaLogManagerDeviceData));
    cudaMemset(md_data, 0, sizeof(CudaLogManagerDeviceData));
  }

  /*!
   * \brief  CudaLogManager Destructor, cleans up device data.
   */
  ~CudaLogManager()
  {
    cudaFree(md_data);
  }

  /*!
   * \brief  Static function that atomically adds up to a maximum value, 
   *         returing the previous value.
   */
  __device__
  static inline logpos_type _atomicAddUpToMax(
      logpos_type *address,
      logpos_type value,
      logpos_type max)
  {
    logpos_type temp =
        *(reinterpret_cast<logpos_type volatile *>(address));
    if (temp < max) {
      logpos_type assumed;
      logpos_type oldval = temp;
      do {
        assumed = oldval;
        oldval = atomicCAS(address, assumed, RAJA_MIN(assumed + value, max));
      } while (assumed != oldval && oldval < max);
      temp = oldval;
    }
    return temp;
  }

  /*!
   * \brief  LogWriter class that facilitates writing logs on the gpu.
   *
   * LogWriter does no error checking.
   * LogWriter requires * sizeof(rw_type) * LogWriter::rwtype_sizeof_msg(...)
   * bytes starting at alignment of alignof(rw_type).
   *
   * The log format used is as follows.
   * rw_type*       - pointer to next log
   *                  (0 or nullptr if this (the current) log does not exist)
   * kernelnum_type - number used for ordering purposes
   * logging_function_type - pointer to a handler function
   * udata_type     - udata argument passed through to handler function
   * print_types[]  - print_types::invalid terminated list of print_types
   *                  indicating the types of the next arguments,
   *                  each padded to sizeof(rw_type)
   * char*|char[], Ts... - fmt string followed by values of arguments,
   *                  each padded to sizeof(rw_type)
   */
  class LogWriter
  {
  public:

    /*!
     * \brief  Static function to determine the size in rw_type needed for 
     *         a message.
     */
    template < typename T_fmt, typename... Ts >
    RAJA_HOST_DEVICE
    static constexpr int
    rwtype_sizeof_msg(kernelnum_type const&, logging_function_type const&,
              udata_type const&, T_fmt const& fmt, Ts const&...)
    {
      return rwtype_sizeof_Ts<rw_type*, kernelnum_type, logging_function_type, udata_type>::value
           + rwtype_sizeof_Ts<print_types[sizeof...(Ts) + 2]>::value
           + rwtype_sizeof_Ts<T_fmt, Ts...>::value;
    }

    /*!
     * \brief  LogWriter constructor that takes pointers into a buffer.
     *
     * Note:   The distance between buf_end and buf_begin must be enough
     *         to hold the message.
     */
    RAJA_HOST_DEVICE
    LogWriter(rw_type*const& buf_begin, rw_type*const& buf_end)
      : m_pos(buf_begin),
        m_end(buf_end)
    {

    }

    /*!
     * \brief  Function that writes a log to the buffer.
     */
    template < typename T_fmt, typename... Ts >
    RAJA_HOST_DEVICE
    void
    write(kernelnum_type const& id, logging_function_type const& func,
          udata_type const& udata, T_fmt const& fmt, Ts const&... args)
    {
      write_value(m_end);
      write_value(id);
      write_value(func);
      write_value(udata);
      write_types_values(fmt, args...);
    }

  private:
    rw_type* m_pos;
    rw_type*const m_end;

    /*!
     * \brief  Function that gets the print_type for its argument.
     *
     * Note:   Unknown types trigger a static_assert.
     */
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< !std::is_arithmetic< T >::value
                             && !std::is_pointer< T >::value
                             && !(std::is_array< T >::value
                                  && std::is_same< char, 
                                        typename std::remove_cv< 
                                            typename std::remove_extent< T >::type
                                                               >::type
                                                 >::value),
                             print_types >::type
    get_type(T const&)
    {
      static_assert(!std::is_same<T, void>::value
                    || std::is_same<T, void>::value,
                    "Raja Logger error: unknown type in logger");
      return print_types::invalid;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value
                             && std::is_signed< T >::value
                             && sizeof(T) == 1,
                             print_types>::type
    get_type(T const&)
    {
      return print_types::int8;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 2, print_types>::type
    get_type(T const&)
    {
      return print_types::int16;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 4, print_types>::type
    get_type(T const&)
    {
      return print_types::int32;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_signed< T >::value && sizeof(T) == 8, print_types>::type
    get_type(T const&)
    {
      return print_types::int64;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 1, print_types>::type
    get_type(T const&)
    {
      return print_types::uint8;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 2, print_types>::type
    get_type(T const&)
    {
      return print_types::uint16;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 4, print_types>::type
    get_type(T const&)
    {
      return print_types::uint32;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_integral< T >::value && std::is_unsigned< T >::value && sizeof(T) == 8, print_types>::type
    get_type(T const&)
    {
      return print_types::uint64;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_floating_point< T >::value && sizeof(T) == 4, print_types>::type
    get_type(T const&)
    {
      return print_types::flt32;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_floating_point< T >::value && sizeof(T) == 8, print_types>::type
    get_type(T const&)
    {
      return print_types::flt64;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_pointer< T >::value
                                && !std::is_same< char, 
                                                  typename std::remove_cv< 
                                                    typename std::remove_pointer< T >::type
                                                  >::type
                                                >::value,
                                    print_types >::type
    get_type(T const&)
    {
      return print_types::ptr;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_pointer< T >::value
                                 && std::is_same< char, 
                                                  typename std::remove_cv< 
                                                    typename std::remove_pointer< T >::type
                                                  >::type
                                                >::value,
                                    print_types >::type
    get_type(T const&)
    {
      return print_types::char_ptr;
    }
    ///
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_array< T >::value
                                 && std::is_same< char, 
                                                  typename std::remove_cv< 
                                                    typename std::remove_extent< T >::type
                                                  >::type
                                                >::value,
                                    print_types >::type
    get_type(T const&)
    {
      return print_types::char_arr;
    }

    /*!
     * \brief  Function that helps write the types array to the buffer.
     */
    RAJA_HOST_DEVICE
    void write_types_hlp(print_types* a)
    {
      a[0] = print_types::invalid;
    }
    ///
    template < typename T0 >
    RAJA_HOST_DEVICE
    void write_types_hlp(print_types* a, T0 const& arg0)
    {
      a[0] = get_type(arg0);
      a[1] = print_types::invalid;
    }
    ///
    template < typename T0, typename T1 >
    RAJA_HOST_DEVICE
    void write_types_hlp(print_types* a, T0 const& arg0, T1 const& arg1)
    {
      a[0] = get_type(arg0);
      a[1] = get_type(arg1);
      a[2] = print_types::invalid;
    }
    ///
    template < typename T0, typename T1, typename T2 >
    RAJA_HOST_DEVICE
    void write_types_hlp(print_types* a, T0 const& arg0, T1 const& arg1, T2 const& arg2)
    {
      a[0] = get_type(arg0);
      a[1] = get_type(arg1);
      a[2] = get_type(arg2);
      a[3] = print_types::invalid;
    }
    ///
    template < typename T0, typename T1, typename T2, typename T3, typename... Ts >
    RAJA_HOST_DEVICE
    void write_types_hlp(print_types* a, T0 const& arg0, T1 const& arg1, T2 const& arg2, T3 const& arg3, Ts const&... args)
    {
      a[0] = get_type(arg0);
      a[1] = get_type(arg1);
      a[2] = get_type(arg2);
      a[3] = get_type(arg3);
      write_types_hlp(a+4, args...);
    }

    /*!
     * \brief  Function that writes the types array to the buffer.
     *
     * Note:   The array is print_types::invalid terminated.
     */
    template < typename... Ts >
    RAJA_HOST_DEVICE
    void write_types_arr(Ts const&... args)
    {
      using pt_arr_type = print_types[sizeof...(Ts) + 1];
      union Trw_union {
        pt_arr_type pt;
        rw_type rw[rwtype_sizeof_Ts<pt_arr_type>::value];
      };

      Trw_union u;
      write_types_hlp(u.pt, args...);

      for (int i = 0; i < rwtype_sizeof_Ts<pt_arr_type>::value; i++) {
        *m_pos++ = u.rw[i];
      }
    }

    /*!
     * \brief  Function that writes the given value to the buffer.
     */
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< !std::is_array< T >::value >::type
    write_value(T const& arg)
    {
      union Trw_union {
        T t;
        rw_type rw[rwtype_sizeof_Ts<T>::value];
      };

      Trw_union u;
      u.t = arg;

      for (int i = 0; i < rwtype_sizeof_Ts<T>::value; i++) {
        *m_pos++ = u.rw[i];
      }
    }

    /*!
     * \brief  Function that writes the given value to the buffer.
     *
     * Note:   Does not record the length of the char[].
     */
    template < typename T >
    RAJA_HOST_DEVICE
    typename std::enable_if< std::is_array< T >::value
                                 && std::is_same< char, 
                                                  typename std::remove_cv< 
                                                    typename std::remove_extent< T >::type
                                                  >::type
                                                >::value >::type
    write_value(T const& arg)
    {
      union Trw_union {
        T t;
        rw_type rw[rwtype_sizeof_Ts<T>::value];
      };

      Trw_union u;
      for( int i = 0; i < std::extent<T>::value; i++ ) {
        u.t[i] = arg[i];
      }

      for ( int i = 0; i < rwtype_sizeof_Ts<T>::value; ++i ) {
        *m_pos++ = u.rw[i];
      }
    }

    /*!
     * \brief  Function that writes the values of the given arguments to 
     *         the buffer.
     */
    RAJA_HOST_DEVICE
    void write_values()
    {

    }
    ///
    template < typename T0, typename... Ts >
    RAJA_HOST_DEVICE
    void write_values(T0 const& arg0, Ts const&... args)
    {
      write_value(arg0);
      write_values(args...);
    }

    /*!
     * \brief  Convenience function that writes the types array and values
     *         for the given arguments to the buffer.
     */
    template < typename... Ts >
    RAJA_HOST_DEVICE
    void write_types_values(Ts const&... args)
    {
      write_types_arr(args...);
      write_values(args...);
    }

  };

  /*!
   * \brief  Class that handles reading a log from the buffer.
   */
  class LogReader;

  /*!
   * \brief  Function that reads the log buffers and clears them.
   */
  void handle_in_order() volatile;

  /*!
   * \brief  function that resets buffers, clearing them of previous state.
   */
  void reset_state() volatile;

  /*!
   * \brief  function that reads a log from the buffer and calls 
   *         the log's handler function.
   */
  static bool
  handle_log(LogReader& reader);

  /*!
   * \brief  function that reads an error from the buffer and calls 
   *         the error's handler function.
   */
  static bool
  handle_error(LogReader& reader);

};

} // closing brace for Internal namespace

/*!
 ******************************************************************************
 *
 * \brief  Specialization of Logger class template for cuda.
 *
 * Note.   Restrictions on cuda log formatting:
 *          - %s  arguments are copied by value, so char* args must point to
 *                 memory accessible on the host when log handlers are called.
 *          - \0  embedded null characters in char arrays and string literals
 *                 are not supported.
 *          - %*d use of * (runtime precision specification) is not currently 
 *                 supported.
 *          - %n  is not supported.
 *
 ******************************************************************************
 */
template < >
class Logger< RAJA::cuda_logger > {
public:
  using func_type = logging_function_type;

  /*!
   * \brief  Constructor for Cuda Logger class template.
   */
  explicit Logger(func_type f = RAJA::basic_logger)
    : m_func(f),
      m_logman(Internal::CudaLogManager::getInstance()),
      m_num(s_kernel_num)
  {

  }

  /*!
   * \brief  Copy constructor for Cuda Logger class template.
   */
  RAJA_HOST_DEVICE
  Logger(Logger< RAJA::cuda_logger > const& other)
    : m_func(other.m_func),
      m_logman(other.m_logman),
#ifdef __CUDA_ARCH__
      m_num(other.m_num)
#else
      m_num(s_kernel_num)
#endif
  {

  }

  /*!
   * \brief  Copy assignment operator deleted.
   */
  Logger& operator=(Logger< RAJA::cuda_logger > const& other) = delete;

  /*!
   * \brief  Cuda Logger class error function.
   */
  template < typename T_fmt, typename... Ts >
  RAJA_HOST_DEVICE
  typename std::enable_if< std::is_pointer< typename std::decay< T_fmt >::type >::value
                        && std::is_same< char, typename std::remove_cv< 
                                                 typename std::remove_pointer< 
                                                   typename std::decay< T_fmt
                                                   >::type
                                                 >::type
                                               >::type
                           >::value
                         >::type
  error(udata_type udata, T_fmt const& fmt, Ts const&... args) const
  {
    m_logman->error_impl(m_num, m_func, udata, fmt, args...);
  }

  /*!
   * \brief  Cuda Logger class log function.
   */
  template < typename T_fmt, typename... Ts >
  RAJA_HOST_DEVICE
  typename std::enable_if< std::is_pointer< typename std::decay< T_fmt >::type >::value
                        && std::is_same< char, typename std::remove_cv< 
                                                 typename std::remove_pointer< 
                                                   typename std::decay< T_fmt
                                                   >::type
                                                 >::type
                                               >::type
                           >::value
                         >::type
  log(udata_type udata, T_fmt const& fmt, Ts const&... args) const
  {
    m_logman->log_impl(m_num, m_func, udata, fmt, args...);
  }

  /*!
   * \brief  Destructor defaulted.
   */
  RAJA_HOST_DEVICE
  ~Logger() = default;

private:
  const func_type m_func = nullptr;
  volatile Internal::CudaLogManager* const m_logman = nullptr;
  const kernelnum_type m_num = s_kernel_num;
};

}  // closing brace for RAJA namespace

#endif

#endif  // closing endif for header file include guard
