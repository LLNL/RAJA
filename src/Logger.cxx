/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Logger functions and classes.
 *
 ******************************************************************************
 */

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

#include "RAJA/RAJA.hxx"

#include <iostream>

namespace RAJA
{

void check_logs()
{
#ifdef RAJA_ENABLE_CUDA
  Internal::CudaLogManager::s_check_logs();
#endif
}

#ifdef RAJA_ENABLE_CUDA
char* Internal::CudaLogManager::s_instance_buffer = nullptr;

loggingID_type Logger< RAJA::cuda_logger >::s_num = 0;

namespace Internal
{

bool s_exit_enabled = true;

class CudaLogManager::LogReader
{
public:
  static const loggingID_type invalid_id = std::numeric_limits<loggingID_type>::max();

  LogReader(LogReader const&) = default;
  LogReader& operator=(LogReader const&) = default;

  LogReader(rw_type*const& buf_pos, rw_type*const& buf_end)
    : m_begin(buf_pos),
      m_pos(buf_pos),
      m_end(buf_end)
  {
    if (m_pos < m_end) {
      read_preamble();
    }
  }

  rw_type* get_next()
  {
    return m_next;
  }

  loggingID_type get_id()
  {
    return m_id;
  }

  logging_function_type get_func()
  {
    return m_func;
  }

  udata_type get_udata()
  {
    return m_udata;
  }

  bool write_msg(std::vector<char>& msg)
  {
    bool err = false;

    const char* fmt_pos = nullptr;
    err = read_type_value(fmt_pos);

    while ( fmt_pos != nullptr ) {

      if (err) {
        break;
      }

      err = do_first_segment(msg, fmt_pos);

    }

    if (m_pos != m_end) {
      fprintf(stderr, "RAJA Logging warning: Unused arguments detected.\n");
    }

    return err;
  }

private:
  rw_type* m_begin = nullptr;
  rw_type* m_pos = nullptr;
  rw_type* m_end = nullptr;

  print_types* m_types = nullptr;

  rw_type* m_next = nullptr;
  loggingID_type m_id = std::numeric_limits<loggingID_type>::max();
  logging_function_type m_func = nullptr;
  udata_type m_udata = -1;

  enum struct width_types : char {
    none,
    hh,
    h,
    l,
    ll,
    j,
    z,
    t,
    L
  };

  bool read_preamble()
  {
    bool err = false;
    err = read_value<rw_type*>(m_next);

    if ( !err && m_next != nullptr ) {
      m_end = m_next;
      err = err || read_value<loggingID_type>(m_id);
      err = err || read_value<logging_function_type>(m_func);
      err = err || read_value<udata_type>(m_udata);
      err = err || skip_types();
    }
    if (err) {
      m_pos = nullptr;
      m_next = nullptr;
      m_id = std::numeric_limits<loggingID_type>::max();
      m_func = nullptr;
      m_udata = -1;
    }
    return err;
  }

  bool skip_types()
  {
    bool err = false;

    m_types = reinterpret_cast<print_types*>(m_pos);

    char* pos = reinterpret_cast<char*>(m_pos);
    char* cend = (char*)memchr(pos, (char)print_types::invalid, sizeof(rw_type)*(m_end - m_pos));
    if (cend != nullptr) {
      int len = (cend - pos) + 1;

      rw_type* end = m_pos + (len + sizeof(rw_type) - 1) / sizeof(rw_type);

      m_pos = end;
    } else {
      fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
      err = true;
      m_pos = m_end;
    }

    return err;
  }

  template < typename T_read, typename T >
  typename std::enable_if< std::is_assignable<T&, T_read>::value, bool >::type
  read_value(T& arg)
  {
    bool err = false;
    union Trw_union {
      T_read t;
      rw_type rw[rwtype_sizeof_Ts<T_read>::value];
    };

    Trw_union u;

    rw_type* end = m_pos + rwtype_sizeof_Ts<T_read>::value;
    if (end <= m_end) {
      for (int i = 0; i < rwtype_sizeof_Ts<T_read>::value; i++) {
        u.rw[i] = m_pos[i];
      }
      m_pos = end;
    } else {
      fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
      u.t = static_cast<T_read>(0);
      err = true;
      m_pos = m_end;
    }

    arg = static_cast<T>(u.t);

    return err;
  }

  template < typename T_read, typename T >
  typename std::enable_if< !std::is_assignable<T&, T_read>::value, bool >::type
  read_value(T& arg)
  {
    bool err = true;

    rw_type* end = m_pos + rwtype_sizeof_Ts<T_read>::value;
    if (end <= m_end) {
      m_pos = end;
    } else {
      fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
      err = true;
      m_pos = m_end;
    }

    fprintf(stderr, "RAJA logger warning: Incompatible types detected.\n");
    arg = static_cast<T>(0);

    return err;
  }

  template < typename T >
  typename std::enable_if< std::is_assignable<T&, char*>::value, bool >::type
  read_value_char_arr(T& arg)
  {
    bool err = false;

    char* pos = reinterpret_cast<char*>(m_pos);
    char* cend = (char*)memchr(pos, '\0', sizeof(rw_type)*(m_end - m_pos));
    if (cend != nullptr) {
      int len = (cend - pos) + 1;

      rw_type* end = m_pos + (len + sizeof(rw_type) - 1) / sizeof(rw_type);

      arg = reinterpret_cast<char*>(m_pos);
      m_pos = end;
    } else {
      fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
      arg = static_cast<T>(0);
      err = true;
      m_pos = m_end;
    }

    return err;
  }

  template < typename T >
  typename std::enable_if< !std::is_assignable<T&, char*>::value, bool >::type
  read_value_char_arr(T& arg)
  {
    bool err = true;

    char* pos = reinterpret_cast<char*>(m_pos);
    char* cend = (char*)memchr(pos, '\0', sizeof(rw_type)*(m_end - m_pos));
    if (cend != nullptr) {
      int len = (cend - pos) + 1;

      rw_type* end = m_pos + (len + sizeof(rw_type) - 1) / sizeof(rw_type);

      m_pos = end;
    } else {
      fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
      err = true;
      m_pos = m_end;
    }

    fprintf(stderr, "RAJA logger warning: Incompatible types detected.\n");
    arg = static_cast<T>(0);

    return err;
  }

  template < typename T >
  bool
  read_type_value(T& arg)
  {
    bool err = false;
    if (m_pos >= m_end || *m_types == print_types::invalid) {
      fprintf(stderr, "RAJA logger warning: No more arguments, Incomplete log detected.\n");
      arg = static_cast<T>(0);
      return err;
    }

    // using print_types = RAJA::Internal::print_types;
    switch( *m_types++ ) {
      case print_types::int8 : {
        err = read_value<int8_t>(arg);
        break;
      }
      case print_types::int16 : {
        err = read_value<int16_t>(arg);
        break;
      }
      case print_types::int32 : {
        err = read_value<int32_t>(arg);
        break;
      }
      case print_types::int64 : {
        err = read_value<int64_t>(arg);
        break;
      }
      case print_types::uint8 : {
        err = read_value<uint8_t>(arg);
        break;
      }
      case print_types::uint16 : {
        err = read_value<uint16_t>(arg);
        break;
      }
      case print_types::uint32 : {
        err = read_value<uint32_t>(arg);
        break;
      }
      case print_types::uint64 : {
        err = read_value<uint64_t>(arg);
        break;
      }
      case print_types::flt32 : {
        err = read_value<float>(arg);
        break;
      }
      case print_types::flt64 : {
        err = read_value<double>(arg);
        break;
      }
      case print_types::ptr : {
        err = read_value<void*>(arg);
        break;
      }
      case print_types::char_ptr : {
        err = read_value<char*>(arg);
        break;
      }
      case print_types::char_arr : {
        err = read_value_char_arr(arg);
        break;
      }
      default : {
        fprintf(stderr, "RAJA logger error: Unknown type encountered.\n");
        err = true;
        break;
      }
    }

    return err;
  }

  template < typename... Ts >
  bool
  vector_sprintf(std::vector<char>& msg, const char*const& fmt, Ts const&... args)
  {
    bool err = false;
    int insert_pos = msg.size() - 1;
    int formatted_size = snprintf(nullptr, 0, fmt, args...);
    if (formatted_size > 0) {
      msg.resize(insert_pos + formatted_size + 1);
      snprintf(&msg[insert_pos], formatted_size + 1, fmt, args...);
    } else if (formatted_size < 0) {
      err = true;
    }
    return err;
  }

  bool
  format_percent(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    return vector_sprintf(msg, fmt);
  }

  bool
  format_char(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        int val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::l : {
        wint_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      default : {
        err = true;
        break;
      }
    }
    return err;
  }

  bool
  format_string(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        char* val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::l : {
        wchar_t* val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      default : {
        err = true;
        break;
      }
    }
    return err;
  }

  bool
  format_signed(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    bool err = false;
    switch (w) {
      case width_types::hh : {
        signed char val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::h : {
        short val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::none : {
        int val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::l : {
        long val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::ll : {
        long long val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::j : {
        intmax_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::z : {
        size_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::t : {
        ptrdiff_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      default : {
        err = true;
        break;
      }
    }
    return err;
  }

  bool
  format_unsigned(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    bool err = false;
    switch (w) {
      case width_types::hh : {
        unsigned char val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::h : {
        unsigned short val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::none : {
        unsigned int val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::l : {
        unsigned long val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::ll : {
        unsigned long long val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::j : {
        uintmax_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::z : {
        size_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::t : {
        ptrdiff_t val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      default : {
        err = true;
        break;
      }
    }
    return err;
  }

  bool
  format_floating_point(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        double val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::l : {
        double val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      case width_types::L : {
        long double val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      default : {
        err = true;
        break;
      }
    }
    return err;
  }

  bool
  format_pointer(std::vector<char>& msg, width_types const& w, const char*const& fmt)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        void* val;
        err = read_type_value(val) || vector_sprintf(msg, fmt, val);
        break;
      }
      default : {
        err = true;
        break;
      }
    }
    return err;
  }

  bool
  get_format_width(width_types& w, const char*const& fmt_begin, const char*const& fmt_end)
  {
    bool err = false;
    w = width_types::none;

    for(const char* pos = fmt_begin; pos < fmt_end; pos++) {
      switch( *pos ) {
        case 'h' : {
          if (w == width_types::none) {
            w = width_types::h;
          } else if (w == width_types::h) {
            w = width_types::hh;
          } else {
            err = true;
          }
          break;
        }
        case 'l' : {
          if (w == width_types::none) {
            w = width_types::l;
          } else if (w == width_types::l) {
            w = width_types::ll;
          } else {
            err = true;
          }
          break;
        }
        case 'j' : {
          if (w == width_types::none) {
            w = width_types::j;
          } else {
            err = true;
          }
          break;
        }
        case 'z' : {
          if (w == width_types::none) {
            w = width_types::z;
          } else {
            err = true;
          }
          break;
        }
        case 't' : {
          if (w == width_types::none) {
            w = width_types::t;
          } else {
            err = true;
          }
          break;
        }
        case 'L' : {
          if (w == width_types::none) {
            w = width_types::L;
          } else {
            err = true;
          }
          break;
        }
        default : {
          break;
        }
      }
    }

    return err;
  }

  bool
  do_single_format(std::vector<char>& msg, const char*& fmt_pos, const char*const& fmt_percent)
  {
    bool err = false;

    const char* fmt_last = strpbrk(fmt_percent+1, "%csdioxXufFeEaAgGnp");
    width_types w;
    err = get_format_width(w, fmt_percent+1, fmt_last);

    if (err || fmt_last == nullptr) {
      fprintf(stderr, "RAJA logger error: Invalid format specifier (%s).\n", fmt_percent);
      err = true;
      return err;
    }

    std::vector<char> fmt_cpy(fmt_last - fmt_pos + 2, '\0');
    memcpy(&fmt_cpy[0], fmt_pos, fmt_last - fmt_pos + 1);

    switch( *fmt_last ) {
      case '%' : {
        err = format_percent(msg, w, &fmt_cpy[0]);
        break;
      }
      case 'c' : {
        err = format_char(msg, w, &fmt_cpy[0]);
        break;
      }
      case 's' : {
        err = format_string(msg, w, &fmt_cpy[0]);
        break;
      }
      case 'd' :
      case 'i' : {
        err = format_signed(msg, w, &fmt_cpy[0]);
        break;
      }
      case 'o' :
      case 'x' :
      case 'X' :
      case 'u' : {
        err = format_unsigned(msg, w, &fmt_cpy[0]);
        break;
      }
      case 'f' :
      case 'F' :
      case 'e' :
      case 'E' :
      case 'a' :
      case 'A' :
      case 'g' :
      case 'G' : {
        err = format_floating_point(msg, w, &fmt_cpy[0]);
        break;
      }
      case 'p' : {
        err = format_pointer(msg, w, &fmt_cpy[0]);
        break;
      }
      case 'n' :
      default : {
        err = true;
        break;
      }
    }
    if (err) {
      fprintf(stderr, "RAJA logger error: Invalid or Unsupported format specifier (%s).\n", &fmt_cpy[fmt_percent - fmt_pos]);
    }

    fmt_pos = fmt_last + 1;

    return err;
  }

  bool
  do_first_segment(std::vector<char>& msg, const char*& fmt_pos)
  {
    bool err = false;

    const char* fmt_percent = strchr(fmt_pos, '%');

    if (fmt_percent != nullptr) {
      // append string up to first %
      err = do_single_format(msg, fmt_pos, fmt_percent);

    } else {
      // done, append remaining string
      err = vector_sprintf(msg, fmt_pos);
      fmt_pos = nullptr;
    }

    return err;
  }

};

bool CudaLogManager::handle_log(LogReader& reader)
{
  // fprintf(stderr, "RAJA logger: handling log.\n");

  // print message to msg buffer.
  std::vector<char> msg(1, '\0');
  bool err = reader.write_msg(msg);

  if ( !err ) {
    logging_function_type func = reader.get_func();
    udata_type udata = reader.get_udata();
    func(udata, &msg[0]);
  } else {
    fprintf(stderr, "RAJA Logging error: Logging function not called.\n");
  }

  return err;
}

bool CudaLogManager::handle_error(LogReader& reader)
{
  ResourceHandler::getInstance().error_cleanup(RAJA::error::user);
  bool err = handle_log(reader);

  if (s_exit_enabled) {
    exit(1);
  }

  return err;
}

void CudaLogManager::handle_in_order() volatile
{
  // wait for logs to be written, ignore cuda errors for now
  cudaDeviceSynchronize();

  {
    // remove volatile from pointers
    rw_type* err_begin = m_error_begin;
    rw_type* err_end   = m_error_pos;

    rw_type* log_begin = m_log_begin;
    rw_type* log_end   = m_log_end;

    LogReader error_reader(err_begin, err_end);
    LogReader log_reader(log_begin, log_end);

    // handle logs in order by id
    while( error_reader.get_id() != LogReader::invalid_id
        || log_reader.get_id()   != LogReader::invalid_id ) {

      if (error_reader.get_id() < log_reader.get_id()) {
        handle_error(error_reader);
        error_reader = LogReader(error_reader.get_next(), err_end);
      } else {
        handle_log(log_reader);
        log_reader = LogReader(log_reader.get_next(), log_end);
      }
    }
  }

  reset_state();
}

void CudaLogManager::reset_state() volatile
{
  // reset buffers and flags
  m_error_pos = m_error_begin;
  memset(m_error_begin, 0, sizeof(rw_type)*(m_error_end - m_error_begin));

  // m_log_pos = m_log_begin;
  memset(m_log_begin, 0, sizeof(rw_type)*(m_log_end - m_log_begin));

  cudaMemset(md_data, 0, sizeof(CudaLogManagerDeviceData));

  m_flag = false;
}

}  // closing brace for Internal namespace

#endif  /* end RAJA_ENABLE_CUDA */

}  // closing brace for RAJA namespace
