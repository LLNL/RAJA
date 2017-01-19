/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Logging functions.
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

namespace Internal
{

  template < typename T_read, typename T >
  typename std::enable_if< std::is_assignable<T&, T_read>::value, bool >::type
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
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
        u.t = static_cast<T_read>(0);
        break;
      }
      u.ca[i] = *pos++;
    }

    arg = static_cast<T>(u.t);

    return err;
  }

  template < typename T_read, typename T >
  typename std::enable_if< !std::is_assignable<T&, T_read>::value, bool >::type
  read_value(char*& pos, char*const& end, T& arg)
  {
    bool err = true;
    for (int i = 0; i < sizeof(T_read); i++) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
        break;
      }
      pos++;
    }

    fprintf(stderr, "RAJA logger warning: Incompatible types detected.\n");
    arg = static_cast<T>(0);

    return err;
  }

  template < typename T >
  typename std::enable_if< std::is_assignable<T&, char*>::value, bool >::type
  read_value_char_arr(char*& pos, char*const& end, T& arg)
  {
    bool err = false;
    arg = static_cast<T>(pos);
    while (!err) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
        arg = static_cast<T>(0);
        break;
      }
      if ( *pos++ == '\0') break;
    }
    return err;
  }

  template < typename T >
  typename std::enable_if< !std::is_assignable<T&, char*>::value, bool >::type
  read_value_char_arr(char*& pos, char*const& end, T& arg)
  {
    bool err = true;
    while (!err) {
      if (pos >= end) {
        fprintf(stderr, "RAJA logger warning: Incomplete log detected.\n");
        break;
      }
      if ( *pos++ == '\0') break;
    }

    fprintf(stderr, "RAJA logger warning: Incompatible types detected.\n");
    arg = static_cast<T>(0);

    return err;
  }

  template < typename T >
  bool
  read_type_value(char*& pos, char*const& end, T& arg)
  {
    bool err = false;
    if (pos >= end) {
      fprintf(stderr, "RAJA logger warning: No more arguments, Incomplete log detected.\n");
      arg = static_cast<T>(0);
      return err;
    }

    // using print_types = RAJA::Internal::print_types;
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
        fprintf(stderr, "RAJA logger error: Unknown type encountered.\n");
        err = true;
        break;
      }
    }

    return err;
  }

  enum struct width_types : char {
    hh,
    h,
    none,
    l,
    ll,
    j,
    z,
    t,
    L
  };

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
  format_char(std::vector<char>& msg, width_types const& w, const char*const& fmt, char*& pos, char*const& end)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        int val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::l : {
        wint_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
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
  format_string(std::vector<char>& msg, width_types const& w, const char*const& fmt, char*& pos, char*const& end)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        char* val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::l : {
        wchar_t* val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
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
  format_signed(std::vector<char>& msg, width_types const& w, const char*const& fmt, char*& pos, char*const& end)
  {
    bool err = false;
    switch (w) {
      case width_types::hh : {
        signed char val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::h : {
        short val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::none : {
        int val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::l : {
        long val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::ll : {
        long long val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::j : {
        intmax_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::z : {
        size_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::t : {
        ptrdiff_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
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
  format_unsigned(std::vector<char>& msg, width_types const& w, const char*const& fmt, char*& pos, char*const& end)
  {
    bool err = false;
    switch (w) {
      case width_types::hh : {
        unsigned char val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::h : {
        unsigned short val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::none : {
        unsigned int val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::l : {
        unsigned long val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::ll : {
        unsigned long long val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::j : {
        uintmax_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::z : {
        size_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::t : {
        ptrdiff_t val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
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
  format_floating_point(std::vector<char>& msg, width_types const& w, const char*const& fmt, char*& pos, char*const& end)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        double val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::l : {
        double val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
        break;
      }
      case width_types::L : {
        long double val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
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
  format_pointer(std::vector<char>& msg, width_types const& w, const char*const& fmt, char*& pos, char*const& end)
  {
    bool err = false;
    switch (w) {
      case width_types::none : {
        void* val;
        read_type_value(pos, end, val);
        err = vector_sprintf(msg, fmt, val) || err;
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
  do_single_format(std::vector<char>& msg, const char*& fmt_pos, const char*const& fmt_percent, char*& pos, char*const& end)
  {
    bool err = false;

    const char* fmt_last = strpbrk(fmt_percent+1, "%csdioxXufFeEaAgGnp");
    width_types w;
    err = get_format_width(w, fmt_percent+1, fmt_last) || err;

    if (err || fmt_last == nullptr) {
      fprintf(stderr, "RAJA logger error: Invalid format specifier (%s).\n", fmt_percent);
      err = true;
      return err;
    }

    std::vector<char> fmt_cpy(fmt_last - fmt_pos + 2, '\0');
    memcpy(&fmt_cpy[0], fmt_pos, fmt_last - fmt_pos + 1);

    switch( *fmt_last ) {
      case '%' : {
        err = format_percent(msg, w, &fmt_cpy[0]) || err;
        break;
      }
      case 'c' : {
        err = format_char(msg, w, &fmt_cpy[0], pos, end) || err;
        break;
      }
      case 's' : {
        err = format_string(msg, w, &fmt_cpy[0], pos, end) || err;
        break;
      }
      case 'd' :
      case 'i' : {
        err = format_signed(msg, w, &fmt_cpy[0], pos, end) || err;
        break;
      }
      case 'o' :
      case 'x' :
      case 'X' :
      case 'u' : {
        err = format_unsigned(msg, w, &fmt_cpy[0], pos, end) || err;
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
        err = format_floating_point(msg, w, &fmt_cpy[0], pos, end) || err;
        break;
      }
      case 'p' : {
        err = format_pointer(msg, w, &fmt_cpy[0], pos, end) || err;
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
  do_first_segment(std::vector<char>& msg, const char*& fmt_pos, char*&pos, char*const& end)
  {
    bool err = false;

    const char* fmt_percent = strchr(fmt_pos, '%');

    if (fmt_percent != nullptr) {
      // append string up to first %
      err = do_single_format(msg, fmt_pos, fmt_percent, pos, end) || err;

    } else {
      // done, append remaining string
      err = vector_sprintf(msg, fmt_pos) || err;
      fmt_pos = nullptr;
    }

    return err;
  }

  bool
  do_printf(std::vector<char>& msg, const char*const& fmt, char*&pos, char*const& end)
  {
    bool err = false;

    const char* fmt_pos = fmt;

    while ( !err && fmt_pos != nullptr ) {

      err = do_first_segment(msg, fmt_pos, pos, end) || err;

    }

    return err;
  }

  bool
  read_preamble(char*& buf_pos, char*& buf_next, char*const& buf_end, loggingID_type& id)
  {
    bool err = false;
    if (buf_pos < buf_next) {
      char* tmp_next;
      err = read_value<char*>(buf_pos, buf_next, tmp_next);
      if ( !err ) {
        buf_next = tmp_next;
        err = read_value<loggingID_type>(buf_pos, buf_next, id);
      }
      if ( err ) {
        buf_pos = buf_end;
        buf_next = buf_end;
        id = std::numeric_limits<loggingID_type>::max();
      }
    } else {
      id = std::numeric_limits<loggingID_type>::max();
    }
    return err;
  }

  bool
  handle_log(char*& buf_pos, char*const& buf_end)
  {
    // fprintf(stderr, "RAJA logger: handling log.\n");
    RAJA::logging_function_type func = nullptr;
    int udata = 0;
    const char* fmt = nullptr;
    std::vector<char> msg(1, '\0');

    // read the logging function, udata, and fmt
    // fprintf(stderr, "RAJA logger: reading function ptr.\n");
    bool err = read_value<RAJA::logging_function_type>(buf_pos, buf_end, func);
    // fprintf(stderr, "RAJA logger: reading udata.\n");
    err = read_value<int>(buf_pos, buf_end, udata) || err;
    // fprintf(stderr, "RAJA logger: reading fmt.\n");
    err = read_type_value(buf_pos, buf_end, fmt) || err;

    if ( !err && fmt != nullptr ) {
      // read the remaining arguments and print to msg buffer.
      err = do_printf(msg, fmt, buf_pos, buf_end) || err;
    }

    if ( !err ) {
      func(udata, &msg[0]);
    } else {
      fprintf(stderr, "RAJA Logging error: Logging function not called.\n");
      buf_pos = buf_end;
    }

    if (buf_pos != buf_end) {
      fprintf(stderr, "RAJA Logging warning: Unused arguments detected.\n");
      buf_pos = buf_end;
    }

    return err;
  }

  bool
  handle_error(char*& buf_pos, char*const& buf_end)
  {
    ResourceHandler::getInstance().error_cleanup(RAJA::error::user);
    bool err = handle_log(buf_pos, buf_end);
    exit(1);
    return err;
  }

  void CudaLogManager::handle_in_order() volatile
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
        error_next = error_end;
        read_preamble(error_pos, error_next, error_end, error_id);
      } else {
        handle_log(log_pos, log_next);
        log_next = log_end;
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

}  // closing brace for Internal namespace

#endif  /* end RAJA_ENABLE_CUDA */

}  // closing brace for RAJA namespace
