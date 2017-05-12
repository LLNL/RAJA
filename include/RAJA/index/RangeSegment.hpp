/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining range segment classes.
 *
 ******************************************************************************
 */

#ifndef RAJA_RangeSegment_HXX
#define RAJA_RangeSegment_HXX

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

#include "RAJA/index/BaseSegment.hpp"
#include "RAJA/internal/Iterators.hpp"

#include <algorithm>
#include <iosfwd>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Segment class representing a contiguous range of indices.
 *
 *         Range is specified by begin and end values.
 *         Traversal executes as:
 *            for (i = m_begin; i < m_end; ++i) {
 *               expression using i as array index.
 *            }
 *
 ******************************************************************************
 */
class RangeSegment : public BaseSegment
{
public:
  ///
  /// Default range segment ctor.
  ///
  /// Segment undefined until begin/end values set.
  ///
  RangeSegment()
      : BaseSegment(_RangeSeg_), m_begin(UndefinedValue), m_end(UndefinedValue)
  {
    ;
  }

  ///
  /// Construct range segment with [begin, end) specified.
  ///
  RangeSegment(Index_type begin, Index_type end)
      : BaseSegment(_RangeSeg_), m_begin(begin), m_end(end)
  {
    ;
  }

  ///
  /// Destructor defined because some compilers don't appear to inline the
  /// one they generate.
  ///
  ~RangeSegment() { ; }

  ///
  /// Copy ctor defined because some compilers don't appear to inline the
  /// one they generate.
  ///
  RangeSegment(const RangeSegment& other)
      : BaseSegment(_RangeSeg_), m_begin(other.m_begin), m_end(other.m_end)
  {
    ;
  }

  ///
  /// Copy assignment operator defined because some compilers don't
  /// appear to inline the one they generate.
  ///
  RangeSegment& operator=(const RangeSegment& rhs)
  {
    if (&rhs != this) {
      RangeSegment copy(rhs);
      this->swap(copy);
    }
    return *this;
  }

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(RangeSegment& other)
  {
    using std::swap;
    swap(m_begin, other.m_begin);
    swap(m_end, other.m_end);
  }

  ///
  /// Return starting index for range.
  ///
  Index_type getBegin() const { return m_begin; }

  ///
  /// Set starting index for range.
  ///
  void setBegin(Index_type begin) { m_begin = begin; }

  ///
  /// Return one past last index for range.
  ///
  Index_type getEnd() const { return m_end; }

  ///
  /// Set one past last index for range.
  ///
  void setEnd(Index_type end) { m_end = end; }

  ///
  /// Return number of indices represented by range.
  ///
  Index_type getLength() const { return (m_end - m_begin); }

  ///
  /// Return 'Owned' indicating that segment object owns the data
  /// representing its indices.
  ///
  IndexOwnership getIndexOwnership() const { return Owned; }

  ///
  /// Equality operator returns true if segments are equal; else false.
  ///
  bool operator==(const RangeSegment& other) const
  {
    return ((m_begin == other.m_begin) && (m_end == other.m_end));
  }

  ///
  /// Inequality operator returns true if segments are not equal, else false.
  ///
  bool operator!=(const RangeSegment& other) const
  {
    return (!(*this == other));
  }

  ///
  /// Equality operator returns true if segments are equal; else false.
  /// (Implements pure virtual method in BaseSegment class).
  ///
  bool operator==(const BaseSegment& other) const
  {
    const RangeSegment* o_ptr = dynamic_cast<const RangeSegment*>(&other);
    if (o_ptr) {
      return (*this == *o_ptr);
    } else {
      return false;
    }
  }

  ///
  /// Inquality operator returns true if segments are not equal; else false.
  /// (Implements pure virtual method in BaseSegment class).
  ///
  bool operator!=(const BaseSegment& other) const
  {
    return (!(*this == other));
  }

  ///
  /// Print segment data to given output stream.
  ///
  void print(std::ostream& os) const;

  using iterator = Iterators::numeric_iterator<Index_type>;

  ///
  /// Get an iterator to the end.
  ///
  iterator end() const { return iterator(m_end); }

  ///
  /// Get an iterator to the beginning.
  ///
  iterator begin() const { return iterator(m_begin); }

  ///
  /// Return the number of elements in the range.
  ///
  Index_type size() const { return m_end - m_begin; }


private:
  Index_type m_begin;
  Index_type m_end;
};

/*!
 ******************************************************************************
 *
 * \brief  Segment class representing a contiguous range of indices with stride.
 *
 *         Range is specified by begin and end values.
 *         Traversal executes as:
 *            for (i = m_begin; i < m_end; i += m_stride) {
 *               expression using i as array index.
 *            }
 *
 ******************************************************************************
 */
class RangeStrideSegment : public BaseSegment
{
public:
  ///
  /// Default range segment with stride ctor.
  ///
  /// Segment undefined until begin/end/stride values set.
  ///
  RangeStrideSegment()
      : BaseSegment(_RangeStrideSeg_),
        m_begin(UndefinedValue),
        m_end(UndefinedValue),
        m_stride(UndefinedValue)
  {
    ;
  }

  ///
  /// Construct range segment [begin, end) and stride specified.
  ///
  RangeStrideSegment(Index_type begin, Index_type end, Index_type stride)
      : BaseSegment(_RangeStrideSeg_),
        m_begin(begin),
        m_end(end),
        m_stride(stride)
  {
    ;
  }

  ///
  /// Destructor defined because some compilers don't appear to inline the
  /// one they generate.
  ///
  ~RangeStrideSegment() { ; }

  ///
  /// Copy ctor defined because some compilers don't appear to inline the
  /// one they generate.
  ///
  RangeStrideSegment(const RangeStrideSegment& other)
      : BaseSegment(_RangeStrideSeg_),
        m_begin(other.m_begin),
        m_end(other.m_end),
        m_stride(other.m_stride)
  {
    ;
  }

  ///
  /// Copy assignment operator defined because some compilers don't
  /// appear to inline the one they generate.
  ///
  RangeStrideSegment& operator=(const RangeStrideSegment& rhs)
  {
    if (&rhs != this) {
      RangeStrideSegment copy(rhs);
      this->swap(copy);
    }
    return *this;
  }

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(RangeStrideSegment& other)
  {
    using std::swap;
    swap(m_begin, other.m_begin);
    swap(m_end, other.m_end);
    swap(m_stride, other.m_stride);
  }

  ///
  /// Return starting index for range.
  ///
  Index_type getBegin() const { return m_begin; }

  ///
  /// Set starting index for range.
  ///
  void setBegin(Index_type begin) { m_begin = begin; }

  ///
  /// Return one past last index for range.
  ///
  Index_type getEnd() const { return m_end; }

  ///
  /// Set one past last index for range.
  ///
  void setEnd(Index_type end) { m_end = end; }

  ///
  /// Return stride for range.
  ///
  Index_type getStride() const { return m_stride; }

  ///
  /// Set stride for range.
  ///
  void setStride(Index_type stride) { m_stride = stride; }

  ///
  /// Return number of indices represented by range.
  ///
  Index_type getLength() const
  {
    return (m_end - m_begin) >= m_stride
               ? (m_end - m_begin) % m_stride ? (m_end - m_begin) / m_stride + 1
                                              : (m_end - m_begin) / m_stride
               : 0;
  }

  ///
  /// Return 'Owned' indicating that segment object owns the data
  /// representing its indices.
  ///
  IndexOwnership getIndexOwnership() const { return Owned; }

  ///
  /// Equality operator returns true if segments are equal; else false.
  ///
  bool operator==(const RangeStrideSegment& other) const
  {
    return ((m_begin == other.m_begin) && (m_end == other.m_end)
            && (m_stride == other.m_stride));
  }

  ///
  /// Inequality operator returns true if segments are not equal, else false.
  ///
  bool operator!=(const RangeStrideSegment& other) const
  {
    return (!(*this == other));
  }

  ///
  /// Equality operator returns true if segments are equal; else false.
  /// (Implements pure virtual method in BaseSegment class).
  ///
  bool operator==(const BaseSegment& other) const
  {
    const RangeStrideSegment* o_ptr =
        dynamic_cast<const RangeStrideSegment*>(&other);
    if (o_ptr) {
      return (*this == *o_ptr);
    } else {
      return false;
    }
  }

  ///
  /// Inquality operator returns true if segments are not equal; else false.
  /// (Implements pure virtual method in BaseSegment class).
  ///
  bool operator!=(const BaseSegment& other) const
  {
    return (!(*this == other));
  }

  ///
  /// Print segment data to given output stream.
  ///
  void print(std::ostream& os) const;

  using iterator = Iterators::strided_numeric_iterator<Index_type>;

  ///
  /// Get an iterator to the end.
  ///
  iterator end() const { return iterator(m_end, m_stride); }

  ///
  /// Get an iterator to the beginning.
  ///
  iterator begin() const { return iterator(m_begin, m_stride); }

  ///
  /// Return the number of elements in the range.
  ///
  Index_type size() const { return getLength(); }

private:
  Index_type m_begin;
  Index_type m_end;
  Index_type m_stride;
};

//
// TODO: Add multi-dim'l ranges, and ability to easily repeat segments using
//       an offset in an index set, others?
//

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
