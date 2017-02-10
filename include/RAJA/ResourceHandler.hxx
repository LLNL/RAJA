/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Functionality to enable callback on raja error.
 *
 ******************************************************************************
 */

#ifndef RAJA_Resource_Handler_HXX
#define RAJA_Resource_Handler_HXX

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
 
#include <vector>
#include <algorithm>

namespace RAJA
{

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
 * \brief  RAJA::check_logs checks if there are logs on the queue and calls the
 *         handlers for what it finds. Synchronize before calling to ensure
 *         handling of all logs.
 */
extern void check_logs();

/*!
 * \brief  Multiple error codes may be returned, use & to get individual errors.
 */
namespace error
{
  enum
  {
    none = 0x0,
    raja = 0x1, // an internal raja error (exceeded allowable reduction varaible count)
    user = 0x2, // an error logged by a user
    cuda = 0x4  // an error caused by cuda
  };
}

/*!
 * \brief  Class that manages function pointers called when RAJA encounters 
 *         an error.
 *
 * Note:   Intended to allow apps to perform things like final logging.
 *         Only called in processes where an error occurs.
 */
class ResourceHandler
{
public:

  /*!
   * \brief  Function pointer type used for the callback.
   */
  using error_cleanup_func = void(*)(int);

  /*!
   * \brief  static funcion to get the ResourceHandler instance.
   */
  static ResourceHandler& getInstance()
  {
    static ResourceHandler me;
    return me;
  }

  /*!
   * \brief  function to add an error cleanup function.
   */
  void add_error_cleanup(error_cleanup_func f)
  {
    m_funcs.push_back(f);
  }

  /*!
   * \brief  function to remove an error cleanup function.
   */
  void remove_error_cleanup(error_cleanup_func f)
  {
    auto loc = std::find(m_funcs.begin(), m_funcs.end(), f);
    if (loc != m_funcs.end()) {
      m_funcs.erase(loc);
    }
  }

  /*!
   * \brief  function that triggers error cleanup callbacks.
   */
  void error_cleanup(int err)
  {
    // call cleanup functions in reverse order
    std::for_each(m_funcs.rbegin(), m_funcs.rend(),
                  [=](error_cleanup_func& f) { f( err ); });
  }

private:

  /*!
   * \brief  private constructor.
   */
  ResourceHandler()
  {

  }

  /*!
   * \brief  private copy functions.
   */
  ResourceHandler(ResourceHandler const&);
  ResourceHandler(ResourceHandler &&);
  ResourceHandler& operator=(ResourceHandler const&);
  ResourceHandler& operator=(ResourceHandler &&);

  /*!
   * \brief  private destructor.
   */
  ~ResourceHandler()
  {

  }

  /*!
   * \brief  container of error callback functions.
   */
  std::vector< error_cleanup_func > m_funcs;
};

/*!
 * \brief  Function to add a callback.
 */
inline void add_cleanup(ResourceHandler::error_cleanup_func f)
{
  ResourceHandler::getInstance().add_error_cleanup(f);
}

/*!
 * \brief  Function to remove a callback.
 */
inline void remove_cleanup(ResourceHandler::error_cleanup_func f)
{
  ResourceHandler::getInstance().remove_error_cleanup(f);
}

}

#endif  // closing endif for header file include guard
