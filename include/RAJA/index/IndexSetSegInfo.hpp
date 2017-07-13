/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining segment info class for index sets.
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSetSegInfo_HPP
#define RAJA_IndexSetSegInfo_HPP

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

#include "RAJA/internal/DepGraphNode.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

class BaseSegment;

/*!
 ******************************************************************************
 *
 * \brief  Class used by IndexSets to hold information about segments.
 *
 *         A segment is defined by its type and its index set object.
 *
 *         The index count value can be provided as a second argument to
 *         forall_Ioff( ) iteration methods to map between actual indices
 *         indices and the running iteration count. That is, the count
 *         for a segment starts with the total length of all segments
 *         preceding that segment.
 *
 *         The dependency graph node member can be used to define a
 *         dependency-graph among segments in an index set.  By default it
 *         is not used.
 *
 ******************************************************************************
 */
class IndexSetSegInfo
{
public:
  ///
  /// Default ctor.
  ///
  IndexSetSegInfo()
      : m_segment(0),
        m_owns_segment(false),
        m_icount(UndefinedValue),
        m_dep_graph_node(0)
  {
    ;
  }

  ///
  /// Ctor to create segment info for give segment.
  ///
  IndexSetSegInfo(BaseSegment* segment, bool owns_segment)
      : m_segment(segment),
        m_owns_segment(owns_segment),
        m_icount(UndefinedValue),
        m_dep_graph_node(0)
  {
    ;
  }

  ~IndexSetSegInfo()
  {
    if (m_dep_graph_node) delete m_dep_graph_node;
  }

  /*
   * Using compiler-provided copy ctor and copy-assignment.
   */

  ///
  /// Retrieve const pointer to base-type segment object.
  ///
  const BaseSegment* getSegment() const { return m_segment; }

  ///
  /// Retrieve pointer to base-type segment object.
  ///
  BaseSegment* getSegment() { return m_segment; }

  ///
  /// Return true if IndexSetSegInfo object owns segment (set at construction);
  /// false, otherwise. False usually means segment is shared by some other
  /// IndexSetSegInfo that owns it.
  ///
  bool ownsSegment() const { return m_owns_segment; }

  ///
  /// Set/get index count for start of segment.
  ///
  void setIcount(Index_type icount) { m_icount = icount; }
  ///
  Index_type getIcount() const { return m_icount; }

  ///
  /// Retrieve const pointer to dependency graph node object for segment.
  ///
  const DepGraphNode* getDepGraphNode() const { return m_dep_graph_node; }

  ///
  /// Retrieve pointer to dependency graph node object for segment.
  ///
  DepGraphNode* getDepGraphNode() { return m_dep_graph_node; }

  ///
  /// Create dependency graph node object for segment and
  /// initialize to default state.
  ///
  /// Note that this method assumes dep node object doesn't already exist.
  ///
  void initDepGraphNode() { m_dep_graph_node = new DepGraphNode(); }

private:
  BaseSegment* m_segment;
  bool m_owns_segment;

  Index_type m_icount;

  DepGraphNode* m_dep_graph_node;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
