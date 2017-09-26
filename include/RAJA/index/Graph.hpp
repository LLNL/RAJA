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

#ifndef RAJA_Graph_HPP
#define RAJA_Graph_HPP

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

#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/RAJAVec.hpp"


namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class representing a dependence graph.
 *
 ******************************************************************************
 */
template <typename StorageT>
class Graph : public RangeSegment
{

public:

  //! Construct graph of size num_vertices = end - begin;
  RAJA_INLINE Graph(StorageT begin, StorageT end) : RangeSegment(begin,end) {
    num_vertices = end - begin;
    starting_vertex = begin;
    vertex_degree.resize(num_vertices);
    vertex_degree_prefix_sum.resize(num_vertices);
    semaphores.resize(num_vertices);
  }

  //! Copy-constructor for Graph ??
  //! Copy-assignment operator for Graph ??

  //! Destroy Graph
  RAJA_INLINE ~Graph() {
  }

  void printGraph(/*const Graph& g, */ std::ostream& output) {
    output << "Graph for Range with starting_vertex="<<starting_vertex
           << ", size="<<num_vertices<<std::endl;

    output << "vertex_degree, size="<<vertex_degree.size()<<": ";
    for (int i=0; i<vertex_degree.size(); i++) {
      output << vertex_degree[i] <<" ";
    } output << std::endl;

    output << "vertex_degree_prefix_sum, size="<<vertex_degree_prefix_sum.size()<<": ";
    for (int i=0; i<vertex_degree_prefix_sum.size(); i++) {
      output << vertex_degree_prefix_sum[i]<<" ";
    } output << std::endl;

    output << "adjacency, size="<<adjacency.size()<<": ";
    for (int i=0; i<adjacency.size(); i++) {
      output << adjacency[i]<<" ";
    } output << std::endl;

    output << "semaphores, size="<<semaphores.size()<<": ";
    for (int i=0; i<semaphores.size(); i++) {
      output << semaphores[i]<<" ";
    } output << std::endl;

  }//end printGraph

  //! Returns the number of segments that depend on me
  RAJA_INLINE
  StorageT getNumDeps(StorageT v) const {
    return vertex_degree[v - starting_vertex];
  }

  //! Returns the number of vertices in the graph
  RAJA_INLINE
  StorageT getNumTasks() const {
    return vertex_degree.size();
  }

  //! Returns the id of the starting task in the graph
  RAJA_INLINE
  StorageT getStartingTask() const {
    return starting_vertex;
  }

  ///
  /// Set semaphore value; i.e., the current number of (unsatisfied)
  /// dependencies that must be satisfied before this task can execute.
  ///
  RAJA_INLINE
  void setSemaphoreValue(StorageT v, StorageT value) {
    RAJA::atomic::atomicExchange<RAJA::atomic::auto_atomic>(&semaphores[v-starting_vertex], value);
  }

  ///
  /// Get semaphore value; i.e., the current number of (unsatisfied)
  /// dependencies that must be satisfied before this task can execute.
  ///
  RAJA_INLINE
  StorageT semaphoreValue(StorageT v) {
    return semaphores[v-starting_vertex];
  }

  ///
  /// Decrement semaphore value; i.e., the current number of (unsatisfied) dependencies
  ///
  RAJA_INLINE
  void decrementSemaphore(StorageT v) {
    RAJA::atomic::atomicDec<RAJA::atomic::auto_atomic>(&semaphores[v-starting_vertex]);
  }

  ///
  /// For all vertices, reset semaphores to be equal to vertex_degree
  /// - could also do this in parallel - would it be faster?
  ///
  RAJA_INLINE
  void resetSemaphores() {
    std::copy(vertex_degree.begin(),vertex_degree.end(),semaphores.begin());
  }

  ///
  /// For vertex v, reset semaphores to be equal to vertex_degree
  ///
  RAJA_INLINE
  void reset(StorageT v) {
    semaphores[v-starting_vertex] = vertex_degree[v-starting_vertex];
  }

  ///
  /// Wait for all dependencies of vertex v to be satisfied
  ///
  void wait(StorageT v) {
    while (semaphores[v-starting_vertex] > 0) {
      // TODO: an efficient wait would be better here, but the standard
      // promise/future is not good enough
      std::this_thread::yield();
    }
  }

  ///
  /// Once the task is completed, set all dependents as ready to go
  ///
  RAJA_INLINE
  void satisfyDependents(StorageT v) {
    std::cout<<"satisfy dependents of v="<<v<<": ";
    int start = 0;
    if (v>0) { start = vertex_degree_prefix_sum[v-starting_vertex-1]; }
    for (int ii = start; ii < vertex_degree_prefix_sum[v-starting_vertex]; ++ii) {
      std::cout<<adjacency[ii]<<" ";
      decrementSemaphore(adjacency[ii]);
    }
  }

  ///
  /// Set up the graph in CSR format
  //
  RAJA_INLINE
  void set_vertex_degree(RAJA::RAJAVec<StorageT>& tmp) {
    std::copy(tmp.begin(),tmp.end(),vertex_degree.begin());
  }
  RAJA_INLINE
  void set_vertex_degree_prefix_sum(RAJA::RAJAVec<StorageT>& tmp) {
    std::copy(tmp.begin(),tmp.end(),vertex_degree_prefix_sum.begin());
  }
  RAJA_INLINE
  void set_adjacency(RAJA::RAJAVec<StorageT>& tmp) {
    adjacency.resize(tmp.size());
    std::copy(tmp.begin(),tmp.end(),adjacency.begin());
  }

protected:

  //! number of vertices in the graph and the starting vertex id
  StorageT num_vertices;
  StorageT starting_vertex;

  /// The graph is stored in CSR format, which stores vertex_degree and corresponding adjacency list

  //! vector vertex_degree - degree for each vertex (one per vertex)
  RAJA::RAJAVec<StorageT> vertex_degree;
  RAJA::RAJAVec<StorageT> vertex_degree_prefix_sum;

  //! vector adjacency - edges for each vertex v, starting at vertex_degree_prefix_sum[v-1] (one per edge)
  RAJA::RAJAVec<StorageT> adjacency;

private:

  //! vector semaphores - number of unsatisfied dependencies (one value per vertex)
  RAJA::RAJAVec<StorageT> semaphores;


};




}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard




//OLGA this is the old OpenMP code
/*

  template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall(
    IndexSet::ExecPolicy<omp_taskgraph_segit, SEG_EXEC_POLICY_T>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
  std::cerr << "\n RAJA IndexSet dependency graph not set , "
  << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));

  int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
    DepGraphNode* task = seg_info->getDepGraphNode();

    task->wait();

    executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

    task->reset();

    if (task->numDepTasks() != 0) {
      for (int ii = 0; ii < task->numDepTasks(); ++ii) {
        // Alternateively, we could get the return value of this call
        // and actively launch the task if we are the last depedent
        // task. In that case, we would not need the semaphore spin
        // loop above.
        int seg = task->depTaskNum(ii);
        DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
        dep->satisfyOne();
      }
    }

  }  // iterate over segments of index set
}


 */
