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

#include <atomic>
#include <cstdlib>
#include <iosfwd>
#include <thread>

#include "RAJA/config.hpp"

#include "RAJA/pattern/atomic.hpp"

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
template <typename StorageT>
class GraphBuilder //: public Graph
{

public:

  RAJA_INLINE GraphBuilder(Graph<StorageT>& _g) : g(_g) {
    num_vertices = g.getNumTasks();
    starting_vertex = g.getStartingTask();
    temp_adjacency.resize(num_vertices);
    vertex_degree.resize(num_vertices);
    vertex_degree_prefix_sum.resize(num_vertices);
    std::fill(vertex_degree.begin(),vertex_degree.end(),0);
  }

  //! Copy-constructor for GraphBuilder ??
  //! Copy-assignment operator for GraphBuilder ??

  //! Destroy GraphBuilder - set up the actual dependence Graph
  RAJA_INLINE ~GraphBuilder() {
    //}

    // If the destructor is not the place to finalize the graph, can have a separate method
    //RAJA_INLINE
    //void finalizeGraph() {

    std::partial_sum(vertex_degree.begin(),vertex_degree.end(),vertex_degree_prefix_sum.begin());
    int adj_size = vertex_degree_prefix_sum[vertex_degree_prefix_sum.size()-1];
    adjacency.resize(adj_size);

    for (int i=0; i<num_vertices; i++) {
      int start = 0;
      if (i>0) { start = vertex_degree_prefix_sum[i-1]; }
      std::copy(temp_adjacency[i].begin(), temp_adjacency[i].end(),
                &(adjacency[start]));
    }

    g.set_vertex_degree(vertex_degree);
    g.set_vertex_degree_prefix_sum(vertex_degree_prefix_sum);
    g.set_adjacency(adjacency);
    g.resetSemaphores();
  }


  ///
  /// Add a vertex and its dependents to the graph
  ///
  void addVertex(StorageT v, std::vector<StorageT>& deps) {
    //put the dependencies into a map; will streamline on graphFinalize
    int num_deps = deps.size();
    vertex_degree[v-starting_vertex] = num_deps;
    temp_adjacency[v-starting_vertex].resize(num_deps);
    std::copy(deps.begin(),deps.end(),temp_adjacency[v-starting_vertex].begin());
  }



protected:

private:

  //! number of vertices in the graph and the starting vertex id
  StorageT num_vertices;
  StorageT starting_vertex;

  /// The graph is stored in CSR format, which stores vertex_degree and corresponding adjacency list

  //! vector vertex_degree - degree for each vertex (one per vertex)
  RAJA::RAJAVec<StorageT> vertex_degree;
  RAJA::RAJAVec<StorageT> vertex_degree_prefix_sum;

  //! vector adjacency - edges for each vertex v, starting at vertex_degree_prefix_sum[v-1] (one per edge)
  RAJA::RAJAVec<StorageT> adjacency;

  //! When building the graph, use a map of adjacent edges per vertex
  // - rewrite in CSR format when GraphBuilder is destroyed
  std::vector<std::vector<StorageT> > temp_adjacency;

  //! the graph we are building
  Graph<StorageT>& g;

};//end GraphBuilder



}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard



