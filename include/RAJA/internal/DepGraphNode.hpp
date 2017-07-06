/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining simple class to manage scheduling
 *          of nodes in a task dependency graph.
 *
 ******************************************************************************
 */

#ifndef RAJA_DepGraphNode_HPP
#define RAJA_DepGraphNode_HPP

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

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include <atomic>
#include <cstdlib>
#include <iosfwd>
#include <thread>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class defining a simple semephore-based data structure for
 *         managing a node in a dependency graph.
 *
 ******************************************************************************
 */
class RAJA_ALIGNED_ATTR(256) DepGraphNode
{
public:
  ///
  /// Constants for total number of allowable dependent tasks
  /// and alignment of semaphore value (should match cache
  /// coherence page size?).
  ///
  /// These values may need to be set differently for different
  /// algorithms and platforms. We haven't determined the best defaults yet!
  ///
  static const int _MaxDepTasks_ = 8;

  ///
  /// Default ctor initializes node to default state.
  ///
  DepGraphNode()
      : m_num_dep_tasks(0), m_semaphore_reload_value(0), m_semaphore_value(0)
  {
  }

  ///
  /// Get/set semaphore value; i.e., the current number of (unsatisfied)
  /// dependencies that must be satisfied before this task can execute.
  ///
  std::atomic<int>& semaphoreValue() { return m_semaphore_value; }

  ///
  /// Get/set semaphore "reload" value; i.e., the total number of external
  /// task dependencies that must be satisfied before this task can execute.
  ///
  int& semaphoreReloadValue() { return m_semaphore_reload_value; }

  ///
  /// Ready this task to be used again
  ///
  void reset() { m_semaphore_value.store(m_semaphore_reload_value); }

  ///
  /// Satisfy one incoming dependency
  ///
  void satisfyOne()
  {
    if (m_semaphore_value > 0) {
      --m_semaphore_value;
    }
  }

  ///
  /// Wait for all dependencies to be satisfied
  ///
  void wait()
  {
    while (m_semaphore_value > 0) {
      // TODO: an efficient wait would be better here, but the standard
      // promise/future is not good enough
      std::this_thread::yield();
    }
  }

  ///
  /// Get/set the number of "forward-dependencies" for this task; i.e., the
  /// number of external tasks that cannot execute until this task completes.
  ///
  int& numDepTasks() { return m_num_dep_tasks; }

  ///
  /// Get/set the forward dependency task number associated with the given
  /// index for this task. This is used to notify the appropriate external
  /// dependencies when this task completes.
  ///
  int& depTaskNum(int tidx) { return m_dep_task[tidx]; }

  ///
  /// Print task graph object node data to given output stream.
  ///
  void print(std::ostream& os) const;

private:
  int m_dep_task[_MaxDepTasks_];
  int m_num_dep_tasks;
  int m_semaphore_reload_value;
  std::atomic<int> m_semaphore_value;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
