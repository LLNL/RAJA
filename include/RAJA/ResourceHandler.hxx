/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   
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

class ResourceHandler
{
public:
  using error_cleanup_func = void(*)(int);

  static ResourceHandler& getInstance()
  {
    static ResourceHandler me;
    return me;
  }

  void add_error_cleanup(error_cleanup_func f)
  {
    m_funcs.push_back(f);
  }

  void remove_error_cleanup(error_cleanup_func f)
  {
    auto loc = std::find(m_funcs.begin(), m_funcs.end(), f);
    if (loc != m_funcs.end()) {
      m_funcs.erase(loc);
    }
  }

  void error_cleanup(int err)
  {
    // call cleanup functions in reverse order
    std::for_each(m_funcs.rbegin(), m_funcs.rend(),
                  [=](error_cleanup_func& f) { f( err ); });
  }

private:

  ResourceHandler()
  {

  }

  ResourceHandler(ResourceHandler const&);
  ResourceHandler(ResourceHandler &&);
  ResourceHandler& operator=(ResourceHandler const&);
  ResourceHandler& operator=(ResourceHandler &&);

  ~ResourceHandler()
  {

  }

  std::vector< error_cleanup_func > m_funcs;
};

}

#endif  // closing endif for header file include guard
