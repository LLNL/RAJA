/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Generic class template for RAJA::Logger.
 *
 ******************************************************************************
 */

#ifndef RAJA_Logger_HXX
#define RAJA_Logger_HXX

#include "RAJA/config.hxx"

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

#include "RAJA/ResourceHandler.hxx"

#include <stdio.h>
#include <stdlib.h>

namespace RAJA
{

/*!
 * \brief  Type for user data given to logger functions.
 */
using udata_type = int;

/*!
 * \brief  Type of logger function handlers.
 */
using logging_function_type = void(*)(udata_type, const char*);

/*!
 * \brief  Type used for identifying and ordering logs.
 */
using loggingID_type = unsigned int;

/*!
 * \brief  RAJA::check_logs checks if there are logs on the queue and calls the
 *         handlers for what it finds. Synchronize before calling to ensure
 *         handling of all logs.
 */
extern void check_logs();

namespace Internal
{
/*!
 * \brief  Static bool that determines if in handling an error RAJA will
 *         call exit.
 *
 * Note:   Use in testing environment only.
 */
extern bool s_exit_enabled;
}

/*!
 * \brief  Default handler function for RAJA::Logger objects.
 */
inline void basic_logger(int udata, const char* msg)
{
  fprintf(stderr, "RAJA log: %s\n", msg);
}

/*!
 ******************************************************************************
 *
 * \brief  Generic Logger class.
 *
 *         A RAJA object that takes a function that handles logging a userdata
 *         and message in a RAJA::forall. The log and error member functions 
 *         enqueue a log, but that log may not be handled immediately.
 *         RAJA::check_logs checks if there are logs on the queue and handles
 *         them in the order in which they arrived.
 *
 *         RAJA::Logger<RAJA::seq_logger> logger(
            [](RAJA::udata_type udata, const char* msg) {
              fprintf(stderr, msg);
              if (udata != 0) {
                my_kill_program();
              }
            });

            RAJA::forall<RAJA::seq_exec>(0, 16, [=](int i){
              if (i > 10) {
                logger.log(0, "Log: found large i = %i", i);
              }
              if (i > 15) {
                logger.error(1, "Error: found too large i = %i", i);
              }
            });
 *
 ******************************************************************************
 */
template < typename policy >
class Logger {
public:

  using func_type = RAJA::logging_function_type;

  /*!
   * \brief  Constructor for Logger taking a logging function handler.
   */
  explicit Logger(func_type f = RAJA::basic_logger)
    : m_func(f)
  {

  }

  /*!
   * \brief  Log member function that formats fmt with args using sprintf and
   *         passes udata and the result to the log handler for this Logger.
   */
  template < typename... T >
  void log(udata_type udata, const char* fmt, T const&... args) const
  {
    int len = snprintf(nullptr, 0, fmt, args...);
    if (len >= 0) {
      char* msg = new char[len+1];
      snprintf(msg, len+1, fmt, args...);
      m_func(udata, msg);
      delete[] msg;
    } else {
      fprintf(stderr, "RAJA logger error: could not format message");
    }
  }

  /*!
   * \brief  Error member function that formats fmt with args using sprintf and
   *         passes udata and the result to the log handler for this Logger.
   *         The error enqueued by this function will cause the program to 
   *         exit when it is handled.
   *
   * Note:   Your error handler function may exit the program, or error will
   *         call exit for you.
   */
  template < typename... T >
  void error(udata_type udata, const char* fmt, T const&... args) const
  {
    ResourceHandler::getInstance().error_cleanup(RAJA::error::user);
    int len = snprintf(nullptr, 0, fmt, args...);
    if (len >= 0) {
      char* msg = new char[len+1];
      snprintf(msg, len+1, fmt, args...);
      m_func(udata, msg);
      delete[] msg;
    } else {
      fprintf(stderr, "RAJA logger error: could not format message");
    }
    if (Internal::s_exit_enabled) {
      exit(1);
    }
  }

private:
  const func_type m_func;
};

}

#endif  // closing endif for header file include guard
