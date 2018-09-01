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

#include <cstdlib>
#include <iosfwd>
#include <iostream>

#include "RAJA/config.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/RAJAVec.hpp"
#include "RAJA/internal/MemUtils_GPU.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class representing a dependence graph.
 *
 ******************************************************************************
 */
template <typename Iterable>
class Graph
{
public:

  //! the underlying value_type type
  using value_type = typename Iterable::value_type;

  //! the underlying iterator type
  using iterator = typename Iterable::iterator;

#if defined(RAJA_ENABLE_CUDA)
  typedef typename RAJA::RAJAVec<value_type, managed_allocator<value_type> > RAJAVec_value_type;
#else
  typedef typename RAJA::RAJAVec<value_type> RAJAVec_value_type;
#endif


  //! Construct graph
  RAJA_INLINE Graph(Iterable& _segment) : segment(_segment),
    m_size(*_segment.end() - *_segment.begin()),
    m_starting_index(*_segment.begin()) {
    m_vertex_degree.resize(m_size);
    m_vertex_degree_prefix_sum.resize(m_size);

    std::cout<<"Constructed graph of size="<<m_size<<", starting_index="<<m_starting_index<<std::endl;

  };


  //! Copy-constructor for Graph ??
  //! Copy-assignment operator for Graph ??

  //! Destroy Graph
  RAJA_INLINE ~Graph() {
  }

  void printGraph(/*const Graph& g, */ std::ostream& output) {
    output << "Graph for Segment with m_starting_index="<<m_starting_index
           << ", size="<<m_size<<std::endl;

    output << "m_vertex_degree, size="<<m_vertex_degree.size()<<": ";
    for (int i=0; i<(int)m_vertex_degree.size(); i++) {
      output << m_vertex_degree[i] <<" ";
    } output << std::endl;

    output << "m_vertex_degree_prefix_sum, size="<<m_vertex_degree_prefix_sum.size()<<": ";
    for (int i=0; i<(int)m_vertex_degree_prefix_sum.size(); i++) {
      output << m_vertex_degree_prefix_sum[i]<<" ";
    } output << std::endl;

    output << "m_adjacency, size="<<m_adjacency.size()<<": ";
    for (int i=0; i<(int)m_adjacency.size(); i++) {
      output << m_adjacency[i]<<" ";
    } output << std::endl;

    output << "m_dependencies, size="<<m_dependencies.size()<<": ";
    for (int i=0; i<(int)m_dependencies.size(); i++) {
      output << m_dependencies[i]<<" ";
    } output << std::endl;

  }//end printGraph

  //! Returns the number of segments that depend on me
  RAJA_INLINE
  value_type getNumDeps(value_type v) const {
    return m_vertex_degree[v - m_starting_index];
  }

  //! Returns the number of vertices in the graph
  RAJA_INLINE
  value_type getNumTasks() const {
    return m_vertex_degree.size();
  }

  //! Returns the id of the starting task in the graph
  RAJA_INLINE
  value_type getStartingTask() const {
    return m_starting_index;
  }

  ///
  /// Set up the graph in CSR format
  ///

  /// For documentation, try to specify which methods are used by GraphBuilder vs. GraphStorage



  RAJA_INLINE
  void copy_vertex_degree(RAJAVec_value_type& tmp) const {
    std::copy(m_vertex_degree.begin(),m_vertex_degree.end(),tmp.begin());
  }
  RAJA_INLINE
  void copy_dependencies(RAJAVec_value_type& tmp) const {
    std::copy(m_dependencies.begin(),m_dependencies.end(),tmp.begin());
  }

  RAJA_INLINE
  value_type get_vertex_degree(const value_type v) const {
    return m_vertex_degree[v-m_starting_index];
  }
  RAJA_INLINE
  value_type get_vertex_degree_prefix_sum(const value_type v) const {
    return m_vertex_degree_prefix_sum[v-m_starting_index];
  }
  RAJA_INLINE
  value_type get_dependencies(const value_type v) const {
    return m_dependencies[v-m_starting_index];
  }
  RAJA_INLINE
  value_type get_adjacency(const value_type v) const {
    return m_adjacency[v];
  }

  RAJA_INLINE
  void set_vertex_degree(RAJAVec_value_type& tmp) {
    std::copy(tmp.begin(),tmp.end(),m_vertex_degree.begin());
  }

  RAJA_INLINE
  void set_vertex_degree_prefix_sum(RAJAVec_value_type& tmp) {
    std::copy(tmp.begin(),tmp.end(),m_vertex_degree_prefix_sum.begin());
  }

  RAJA_INLINE
  void set_dependencies(RAJAVec_value_type& tmp) {
    m_dependencies.resize(tmp.size());
    std::copy(tmp.begin(),tmp.end(),m_dependencies.begin());
  }

  RAJA_INLINE
  void set_adjacency(RAJAVec_value_type& tmp) {
    m_adjacency.resize(tmp.size());
    std::copy(tmp.begin(),tmp.end(),m_adjacency.begin());
  }

  RAJA_INLINE
  value_type get_starting_index() const {
    return m_starting_index;
  }
  //! obtain an iterator to the beginning of the underlying segment
  /*!
   * \return an iterator corresponding to the beginning of the Segment
   */
  RAJA_HOST_DEVICE iterator begin() const { return segment.begin(); }

  //! obtain an iterator to the end of the underlying segment
  /*!
   * \return an iterator corresponding to the end of the Segment
   */
  RAJA_HOST_DEVICE iterator end() const { return segment.end(); }

  //! obtain the size of the underlying segment
  /*!
   * the size is calculated by determing the actual trip count in the
   * interval of [begin, end) with a specified step
   *
   * \return the total number of steps for this Segment
   */
  //RAJA_HOST_DEVICE value_type size() const { return segment.size(); }
  RAJA_INLINE
  value_type size() { return segment.size(); }

  RAJA_INLINE
  value_type size() const { return segment.size(); }

protected:
  //! The range
  Iterable segment;

  //! number of vertices in the graph and the starting vertex id
  value_type m_size;
  value_type m_starting_index;

  /// The graph is stored in CSR format, which stores m_vertex_degree and corresponding m_adjacency list

  //! vector m_vertex_degree - degree for each vertex (one per vertex)
  RAJAVec_value_type m_vertex_degree;
  RAJAVec_value_type m_vertex_degree_prefix_sum;
  RAJAVec_value_type m_dependencies;

  //! vector m_adjacency - edges for each vertex v, starting at m_vertex_degree_prefix_sum[v-1] (one per edge)
  RAJAVec_value_type m_adjacency;

private:


};


//! Alias for Graph<RangeSegment>
using GraphRangeSegment = Graph<RangeSegment>;




}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
