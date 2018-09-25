/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining class to manage scheduling
 *          of nodes in a task dependency graph.
 *
 ******************************************************************************
 */

#ifndef RAJA_GraphBuilder_HPP
#define RAJA_GraphBuilder_HPP

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

#include <cstdlib>
#include <iosfwd>
#include <thread>
#include <numeric>

#include "RAJA/config.hpp"

#include "RAJA/index/Graph.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/RAJAVec.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class representing a dependence graph builder
 *
 ******************************************************************************
 */
//template <typename value_type>
template <typename P> // P = Graph<value_type>
class GraphBuilder //: public Graph
{
public:

  using value_type = typename P::value_type;

#if defined(RAJA_ENABLE_CUDA)
  typedef typename RAJA::RAJAVec<value_type, managed_allocator<value_type> > RAJAVec_value_type;
#else
  typedef typename RAJA::RAJAVec<value_type> RAJAVec_value_type;
#endif

  RAJA_INLINE GraphBuilder(Graph<P>& _g) : g(_g) {
    m_size = g.getNumTasks();
    m_starting_vertex = g.getStartingTask();
    m_temp_adjacency.resize(m_size);
    m_vertex_degree.resize(m_size);
    m_vertex_degree_prefix_sum.resize(m_size);
    m_dependencies.resize(m_size);
    std::fill(m_vertex_degree.begin(),m_vertex_degree.end(),0);
    std::fill(m_dependencies.begin(),m_dependencies.end(),0);
  }

  //! Copy-constructor for GraphBuilder ??
  //! Copy-assignment operator for GraphBuilder ??

  //! Destructor
  RAJA_INLINE ~GraphBuilder() {
  }

  ///
  /// Create the CSR format graph
  ///
  void createDependenceGraph() {

    std::partial_sum(m_vertex_degree.begin(),m_vertex_degree.end(),m_vertex_degree_prefix_sum.begin());

    int adj_size = m_vertex_degree_prefix_sum[m_vertex_degree_prefix_sum.size()-1];
    m_adjacency.resize(adj_size);

    for (int i=0; i<m_size; i++) {
      int start = 0;
      if (i>0) { start = m_vertex_degree_prefix_sum[i-1]; }
      std::copy(m_temp_adjacency[i].begin(), m_temp_adjacency[i].end(),
                &(m_adjacency[start]));
    }

    g.set_vertex_degree(m_vertex_degree);
    g.set_vertex_degree_prefix_sum(m_vertex_degree_prefix_sum);
    g.set_dependencies(m_dependencies);
    g.set_adjacency(m_adjacency);
  }


  ///
  /// Add a vertex and its dependents to the graph
  ///
  void addVertex(value_type v, std::vector<value_type>& deps) {
    //put the dependencies into a map; will streamline on graphFinalize
    int num_deps = deps.size();
    m_vertex_degree[v-m_starting_vertex] = num_deps;
    m_temp_adjacency[v-m_starting_vertex].resize(num_deps);
    std::copy(deps.begin(),deps.end(),m_temp_adjacency[v-m_starting_vertex].begin());

    for (typename std::vector<value_type>::iterator it = deps.begin(); it != deps.end(); ++it) {
      m_dependencies[*it-m_starting_vertex]++;
    }
  }



protected:

private:

  //! number of vertices in the graph and the starting vertex id
  value_type m_size;
  value_type m_starting_vertex;

  /// The graph is stored in CSR format, which stores m_vertex_degree and corresponding m_adjacency list

  //! vector m_vertex_degree - degree for each vertex (one per vertex)
  RAJAVec_value_type m_vertex_degree;
  RAJAVec_value_type m_vertex_degree_prefix_sum;
  RAJAVec_value_type m_dependencies;

  //! vector m_adjacency - edges for each vertex v, starting at m_vertex_degree_prefix_sum[v-1] (one per edge)
  RAJAVec_value_type m_adjacency;


  //! When building the graph, use a map of adjacent edges per vertex
  // - rewrite in CSR format when GraphBuilder is destroyed
#if defined(RAJA_ENABLE_CUDA)
  std::vector<std::vector<value_type, managed_allocator<value_type> >, managed_allocator<value_type> > m_temp_adjacency;
#else
  std::vector<std::vector<value_type> > m_temp_adjacency;
#endif

  //! the graph we are building
  Graph<P>& g;

};//end GraphBuilder



}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
