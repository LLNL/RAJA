// /*!
//  ******************************************************************************
//  *
//  * \file
//  *
//  * \brief   RAJA header file defining class to manage scheduling
//  *          of nodes in a task dependency graph.
//  *
//  ******************************************************************************
//  */

 #ifndef RAJA_GraphStorage_HPP
 #define RAJA_GraphStorage_HPP

// //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// // Copyright (c) 2016, Lawrence Livermore National Security, LLC.
// //
// // Produced at the Lawrence Livermore National Laboratory
// //
// // LLNL-CODE-689114
// //
// // All rights reserved.
// //
// // This file is part of RAJA.
// //
// // For additional details, please also read RAJA/LICENSE.
// //
// // Redistribution and use in source and binary forms, with or without
// // modification, are permitted provided that the following conditions are met:
// //
// // * Redistributions of source code must retain the above copyright notice,
// //   this list of conditions and the disclaimer below.
// //
// // * Redistributions in binary form must reproduce the above copyright notice,
// //   this list of conditions and the disclaimer (as noted below) in the
// //   documentation and/or other materials provided with the distribution.
// //
// // * Neither the name of the LLNS/LLNL nor the names of its contributors may
// //   be used to endorse or promote products derived from this software without
// //   specific prior written permission.
// //
// // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// // ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// // LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// // DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// // DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// // OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// // HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// // STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// // IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// // POSSIBILITY OF SUCH DAMAGE.
// //
// //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <atomic>
#include <cstdlib>
#include <iosfwd>
#include <thread>
#include <iostream>

#include "RAJA/config.hpp"
#include "RAJA/pattern/atomic.hpp"
#include "RAJA/index/Graph.hpp"

namespace RAJA
{

// /*!
//  ******************************************************************************
//  *
//  * \brief  Class representing a dependence graph.
//  *
//  ******************************************************************************
//  */
template <typename Iterable>
class GraphStorage
{
public:

  //! the underlying value_type type
  using value_type = typename Iterable::value_type;

  //! the underlying iterator type
  using iterator = typename Iterable::iterator;

  //! Construct graph storage
  RAJA_INLINE GraphStorage(const Iterable& _g) : g(_g),
                                                 m_size(g.size()),
                                                 m_starting_index(g.get_starting_index()) {
    m_semaphores.resize(m_size);
    reset();
  };


  //! Copy-constructor for GraphStorage ??
  //! Copy-assignment operator for GraphStorage ??

  //! Destroy GraphStorage
  RAJA_INLINE ~GraphStorage() {
  }

  void printGraph(/*const GraphStorage& g, */ std::ostream& output) {
    output << "GraphStorage for Range with starting_index="<<m_starting_index
           << ", size="<<g.size()
           << ", m_semaphores: ";

    for (int i=0; i<m_semaphores.size(); i++) {
      output << m_semaphores[i]<<" ";
    } output << std::endl;

  }//end printGraph

  ///
  /// Set semaphore value; i.e., the current number of (unsatisfied)
  /// dependencies that must be satisfied before this task can execute.
  ///
  RAJA_INLINE
  void setSemaphoreValue(const value_type v, const value_type value) {
    RAJA::atomic::atomicExchange<RAJA::atomic::auto_atomic>(&m_semaphores[v-m_starting_index], value);
  }

  ///
  /// Get semaphore value; i.e., the current number of (unsatisfied)
  /// dependencies that must be satisfied before this task can execute.
  ///
  RAJA_INLINE
  value_type getSemaphoreValue(const value_type v) {
    return m_semaphores[v-m_starting_index];
  }

  ///
  /// Decrement semaphore value; i.e., the current number of (unsatisfied) dependencies
  ///
  RAJA_INLINE
  void decrementSemaphore(const value_type v) {
    RAJA::atomic::atomicDec<RAJA::atomic::auto_atomic>(&m_semaphores[v-m_starting_index]);
#if 0
    if (m_semaphores[v-m_starting_index] < 0) {
      std::cout<<v<<"| ERROR NEGATIVE SEMAPHORE"<<std::endl;
    }
#endif

  }

  ///
  /// For all vertices, reset m_semaphores to be equal to vertex_degree
  /// - could also do this in parallel - would it be faster?
  ///
  RAJA_INLINE
  void reset() {
    g.copy_dependencies(m_semaphores);
  }

  ///
  /// For vertex v, reset m_semaphores to be equal to vertex_degree
  ///
  RAJA_INLINE
  void reset(const value_type v) {
    setSemaphoreValue(v,g.get_vertex_degree(v));
  }

  ///
  /// For vertex v, yield control while semaphore > 0
  /// TODO: an efficient wait would be better here, but the standard
  ///       promise/future is not good enough
  ///
  void wait(const value_type v) {
    while (m_semaphores[v-m_starting_index] > 0) {
      std::this_thread::yield();
    }
  }

  ///
  /// Once the task is completed, decrement semaphores of all dependents
  ///
  RAJA_INLINE
  void completed(const value_type v) {
    int start = 0;
    int end   = g.get_vertex_degree_prefix_sum(v);
    if (v-m_starting_index > 0) { start = g.get_vertex_degree_prefix_sum(v-1); }

    for (int ii = start; ii < end; ii++) {
      decrementSemaphore(g.get_adjacency(ii));
    }
  }

protected:

private:
  //! number of vertices in the graph and the starting vertex id
  const Iterable& g;
  value_type m_size;
  value_type m_starting_index;

  //! vector m_semaphores - number of unsatisfied dependencies (one value per vertex)
#if defined(RAJA_ENABLE_CUDA)
  RAJA::RAJAVec<value_type, managed_allocator<value_type> > m_semaphores;
#else
  RAJA::RAJAVec<value_type> m_semaphores;
#endif

};


//! Alias for GraphStorage
using GraphStorageRange = GraphStorage<GraphRangeSegment>;



}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard

