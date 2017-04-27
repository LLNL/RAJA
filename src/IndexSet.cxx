/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for index set classes
 *
 ******************************************************************************
 */

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

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include <iostream>

namespace RAJA
{

/*
*************************************************************************
*
* Public IndexSet methods for basic object mechanics.
*
*************************************************************************
*/

IndexSet::IndexSet()
    : m_len(0), m_segments(), m_private(0), m_dep_graph_set(false)
{
}

IndexSet::IndexSet(const IndexSet& other)
    : m_len(0), m_segments(), m_private(0), m_dep_graph_set(false)
{
  copy(other);
}

IndexSet& IndexSet::operator=(const IndexSet& rhs)
{
  if (&rhs != this) {
    IndexSet copy(rhs);
    this->swap(copy);
  }
  return *this;
}

IndexSet::~IndexSet()
{
  for (size_t i = 0; i < m_segments.size(); ++i) {
    IndexSetSegInfo& seg_info = m_segments[i];

    if (seg_info.ownsSegment()) {
      delete seg_info.getSegment();
    }
  }
}

void IndexSet::swap(IndexSet& other)
{
  using std::swap;
  swap(m_len, other.m_len);
  swap(m_segments, other.m_segments);
  swap(m_private, other.m_private);
  swap(m_dep_graph_set, other.m_dep_graph_set);
}

/*
*************************************************************************
*
* Method to check whether given segment is a valid type for this
* IndexSet class.
*
*************************************************************************
*/
bool IndexSet::isValidSegmentType(const BaseSegment* segment) const
{
  bool ret_val = false;

  if (segment) {
    SegmentType seg_type = segment->getType();

    if (seg_type == _RangeSeg_ ||
#if 0  // RDH RETHINK
           seg_type == _RangeStrideSeg_ ||
#endif
        seg_type == _ListSeg_) {
      ret_val = true;
    }
  }

  return ret_val;
}

/*
*************************************************************************
*
* Methods to add segments to index set.
*
*************************************************************************
*/

bool IndexSet::push_back(const BaseSegment& segment)
{
  bool retval = false;

  if (isValidSegmentType(&segment)) {
    BaseSegment* new_seg = createSegmentCopy(segment);
    retval = push_back_private(new_seg, true /* owns segment */);
    if (!retval) delete new_seg;
  }

  return retval;
}

bool IndexSet::push_front(const BaseSegment& segment)
{
  bool retval = false;

  if (isValidSegmentType(&segment)) {
    BaseSegment* new_seg = createSegmentCopy(segment);
    retval = push_front_private(new_seg, true /* owns segment */);
    if (!retval) delete new_seg;
  }

  return retval;
}

bool IndexSet::push_back(BaseSegment&& segment)
{
  bool retval = false;

  if (isValidSegmentType(&segment)) {
    BaseSegment* new_seg = createSegmentCopy(std::move(segment));
    retval = push_back_private(new_seg, true /* owns segment */);
    if (!retval) delete new_seg;
  }

  return retval;
}
bool IndexSet::push_front(BaseSegment&& segment)
{
  bool retval = false;

  if (isValidSegmentType(&segment)) {
    BaseSegment* new_seg = createSegmentCopy(std::move(segment));
    retval = push_front_private(new_seg, true /* owns segment */);
    if (!retval) delete new_seg;
  }

  return retval;
}

/*
*************************************************************************
*
* Methods to create IndexSet "views".
*
*************************************************************************
*/

IndexSet* IndexSet::createView(int begin, int end) const
{
  IndexSet* retVal = new IndexSet();

  int numSeg = m_segments.size();
  int minSeg = ((begin >= 0) ? begin : 0);
  int maxSeg = ((end < numSeg) ? end : numSeg);
  for (int i = minSeg; i < maxSeg; ++i) {
    retVal->push_back_nocopy(
        const_cast<BaseSegment*>(m_segments[i].getSegment()));
  }

  return retVal;
}

IndexSet* IndexSet::createView(const int* segIds, int len) const
{
  IndexSet* retVal = new IndexSet();

  int numSeg = m_segments.size();
  for (int i = 0; i < len; ++i) {
    if (segIds[i] >= 0 && segIds[i] < numSeg) {
      retVal->push_back_nocopy(
          const_cast<BaseSegment*>(m_segments[segIds[i]].getSegment()));
    }
  }

  return retVal;
}

/*
*************************************************************************
*
* Set [begin, end) interval of segment ids.
*
*************************************************************************
*/
void IndexSet::setSegmentInterval(size_t interval_id, int begin, int end)
{
  while (m_seg_interval_begin.size() < interval_id + 1) {
    m_seg_interval_begin.push_back(UndefinedValue);
    m_seg_interval_end.push_back(UndefinedValue);
  }
  m_seg_interval_begin[interval_id] = begin;
  m_seg_interval_end[interval_id] = end;
}

/*
*************************************************************************
*
* Create dependency graph node objects (with default state) for segments.
*
*************************************************************************
*/
void IndexSet::initDependencyGraph()
{
  for (size_t i = 0; i < m_segments.size(); ++i) {
    IndexSetSegInfo& seg_info = m_segments[i];
    seg_info.initDepGraphNode();
  }
}

/*
*************************************************************************
*
* Index equality check method.
*
*************************************************************************
*/
bool IndexSet::operator==(const IndexSet& other) const
{
  // Are # segments the same, as well as total length??
  bool equal = (m_segments.size() == other.m_segments.size()
                && getLength() == other.getLength());

  if (equal) {
    size_t isi = 0;
    while (equal && isi < m_segments.size()) {
      const BaseSegment* iseg = getSegmentInfo(isi)->getSegment();
      const BaseSegment* o_iseg = other.getSegmentInfo(isi)->getSegment();

      equal = (*iseg == *o_iseg);

      isi++;
    }
  }

  return equal;
}

/*
*************************************************************************
*
* Print contents of index set to given output stream.
*
*************************************************************************
*/
void IndexSet::print(std::ostream& os) const
{
  os << "\nINDEX SET : "
     << " length = " << getLength() << std::endl
     << "      num segments = " << getNumSegments() << std::endl
     << "      dependency graph set = " << dependencyGraphSet() << std::endl;

  for (size_t isi = 0; isi < m_segments.size(); ++isi) {
    os << "\nSegment # " << isi << " : " << std::endl;

    const IndexSetSegInfo* seg_info = getSegmentInfo(isi);

    const BaseSegment* iseg = seg_info->getSegment();
    SegmentType segtype = iseg->getType();

    switch (segtype) {
      case _RangeSeg_: {
        if (iseg) {
          os << "\t icount = " << seg_info->getIcount() << std::endl;
          static_cast<const RangeSegment*>(iseg)->print(os);
        } else {
          os << "_RangeSeg_ is null" << std::endl;
        }
        break;
      }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            if ( iseg ) {
               os << "\t icount = " << seg_info->getIcount() << std::endl;
               static_cast<const RangeStrideSegment*>(iseg)->print(os);
            } else {
               os << "_RangeStrideSeg_ is null" << std::endl;
            }
            break;
         }
#endif

      case _ListSeg_: {
        if (iseg) {
          os << "\t icount = " << seg_info->getIcount() << std::endl;
          static_cast<const ListSegment*>(iseg)->print(os);
        } else {
          os << "_ListSeg_ is null" << std::endl;
        }
        break;
      }

      default: {
        os << "IndexSet print: case not implemented!!\n";
      }

    }  // switch ( segtype )

    const DepGraphNode* task = seg_info->getDepGraphNode();
    if (task) {
      task->print(os);
    }

  }  // iterate over segments...
}

/*
*************************************************************************
*
* Private helper function to copy index set segments.
*
* Note: Assumes this index set is empty.
*
*************************************************************************
*/
void IndexSet::copy(const IndexSet& other)
{
  const int num_segs = other.getNumSegments();
  for (int isi = 0; isi < num_segs; ++isi) {
    const BaseSegment* iseg = other.getSegment(isi);
    SegmentType segtype = iseg->getType();

    if (iseg) {
      switch (segtype) {
        case _RangeSeg_: {
          push_back(*static_cast<const RangeSegment*>(iseg));
          break;
        }

#if 0  // RDH RETHINK
            case _RangeStrideSeg_ : {
               push_back(*static_cast<const RangeStrideSegment*>(iseg));
               break;
            }
#endif

        case _ListSeg_: {
          push_back(*static_cast<const ListSegment*>(iseg));
          break;
        }

        default: {
          std::cout << "\t IndexSet::copy: case not implemented!!\n";
        }

      }  // switch ( segtype )

    }  // if ( iset )

  }  // for isi...
}

/*
*************************************************************************
*
* Private helper methods to add segments to index set.
*
*************************************************************************
*/

bool IndexSet::push_back_private(BaseSegment* seg, bool owns_segment)
{
  if (isValidSegmentType(seg)) {
    m_segments.push_back(IndexSetSegInfo(seg, owns_segment));
    m_segments[m_segments.size() - 1].setIcount(m_len);

    m_len += seg->getLength();

    return true;

  } else {
    return false;
  }
}

bool IndexSet::push_front_private(BaseSegment* seg, bool owns_segment)
{
  if (isValidSegmentType(seg)) {
    m_segments.push_front(IndexSetSegInfo(seg, owns_segment));
    m_segments[0].setIcount(0);
    m_len += seg->getLength();

    Index_type icount = seg->getLength();
    for (size_t i = 1; i < m_segments.size(); ++i) {
      IndexSetSegInfo& seg_info = m_segments[i];

      seg_info.setIcount(icount);

      icount += seg_info.getSegment()->getLength();
    }

    return true;

  } else {
    return false;
  }
}

BaseSegment* IndexSet::createSegmentCopy(const BaseSegment& segment) const
{
  BaseSegment* new_seg = 0;

  switch (segment.getType()) {
    case _RangeSeg_: {
      const RangeSegment& seg = static_cast<const RangeSegment&>(segment);
      new_seg = new RangeSegment(seg);
      break;
    }

#if 0  // RDH RETHINK
      case _RangeStrideSeg_ : {
         const RangeStrideSegment& seg = static_cast<const RangeStrideSegment&>(segment);
         new_seg = new RangeStrideSegment(seg); 
         break;
      }
#endif

    case _ListSeg_: {
      const ListSegment& seg = static_cast<const ListSegment&>(segment);
      new_seg = new ListSegment(seg);
      break;
    }

    default: {
    }

  }  // switch ( segtype )

  return new_seg;
}

BaseSegment* IndexSet::createSegmentCopy(BaseSegment&& segment) const
{
  BaseSegment* new_seg = 0;

  switch (segment.getType()) {
    case _RangeSeg_: {
      RangeSegment&& seg = static_cast<RangeSegment&&>(segment);
      new_seg = new RangeSegment(std::move(seg));
      break;
    }

#if 0  // RDH RETHINK
      case _RangeStrideSeg_ : {
         RangeStrideSegment&& seg = static_cast<RangeStrideSegment&&>(segment);
         new_seg = new RangeStrideSegment(std::move(seg)); 
         break;
      }
#endif

    case _ListSeg_: {
      ListSegment&& seg = static_cast<ListSegment&&>(segment);
      new_seg = new ListSegment(std::move(seg));
      break;
    }

    default: {
    }

  }  // switch ( segtype )

  return new_seg;
}

}  // closing brace for RAJA namespace
