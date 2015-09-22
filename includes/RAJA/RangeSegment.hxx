/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining range segment classes.
 *     
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_RangeSegment_HXX
#define RAJA_RangeSegment_HXX

#include "BaseSegment.hxx"

#include "execpolicy.hxx"

#include <iosfwd>


namespace RAJA {


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
   /// Sequential execution policy for range segment.
   ///
   typedef RAJA::seq_exec seq_policy;

   ///
   /// Default range segment ctor.
   ///
   /// Segment undefined until begin/end values set.
   ///
   RangeSegment()
   : BaseSegment( _RangeSeg_ ), 
     m_begin(UndefinedValue), 
     m_end(UndefinedValue) { ; }

   ///
   /// Construct range segment with [begin, end) specified.
   ///
   RangeSegment(Index_type begin, Index_type end) 
   : BaseSegment( _RangeSeg_ ), 
     m_begin(begin), 
     m_end(end) { ; }

   /*
    * Using compiler-generated dtor, copy ctor, copy assignment.
    */

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
   Index_type getLength() const { return (m_end-m_begin); }

   ///
   /// Return 'Owned' indicating that segment object owns the data
   /// representing its indices.
   ///
   IndexOwnership getIndexOwnership() const { return Owned; }

   ///
   /// Equality operator returns true if segments are equal; else false.
   ///
   bool operator ==(const RangeSegment& other) const
   {
      return ( (m_begin == other.m_begin) && (m_end == other.m_end) );
   }

   ///
   /// Inequality operator returns true if segments are not equal, else false.
   ///
   bool operator !=(const RangeSegment& other) const
   {
      return ( !(*this == other) );
   }

   ///
   /// Equality operator returns true if segments are equal; else false.
   /// (Implements pure virtual method in BaseSegment class).
   ///
   bool operator ==(const BaseSegment& other) const
   {
      const RangeSegment* o_ptr = dynamic_cast<const RangeSegment*>(&other);
      if ( o_ptr ) {
        return ( *this == *o_ptr );
      } else {
        return false;
      }
   }

   ///
   /// Inquality operator returns true if segments are not equal; else false.
   /// (Implements pure virtual method in BaseSegment class).
   ///
   bool operator !=(const BaseSegment& other) const
   {
      return ( !(*this == other) );
   }

   ///
   /// Print segment data to given output stream.
   ///
   void print(std::ostream& os) const;

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
   /// Sequential execution policy for range segment with stride.
   ///
   typedef RAJA::seq_exec seq_policy;

   ///
   /// Default range segment with stride ctor.
   ///
   /// Segment undefined until begin/end/stride values set.
   ///
   RangeStrideSegment()
   : BaseSegment( _RangeStrideSeg_ ),
     m_begin(UndefinedValue), 
     m_end(UndefinedValue), 
     m_stride(UndefinedValue) { ; }

   ///
   /// Construct range segment [begin, end) and stride specified.
   ///
   RangeStrideSegment(Index_type begin, Index_type end, Index_type stride)
   : BaseSegment( _RangeStrideSeg_ ), 
     m_begin(begin), 
     m_end(end), 
     m_stride(stride) { ; }

   /*
    * Using compiler-generated dtor, copy ctor, copy assignment.
    */

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
   Index_type getLength() const { return (m_end-m_begin) >= m_stride ?
                                         (m_end-m_begin)/m_stride + 1 : 0; }

   ///
   /// Return 'Owned' indicating that segment object owns the data
   /// representing its indices.
   ///
   IndexOwnership getIndexOwnership() const { return Owned; }

   ///
   /// Equality operator returns true if segments are equal; else false.
   ///
   bool operator ==(const RangeStrideSegment& other) const
   {
      return ( (m_begin == other.m_begin) && 
               (m_end == other.m_end) &&
               (m_stride == other.m_stride) );
   }

   ///
   /// Inequality operator returns true if segments are not equal, else false.
   ///
   bool operator !=(const RangeStrideSegment& other) const
   {
      return ( !(*this == other) );
   }

   ///
   /// Equality operator returns true if segments are equal; else false.
   /// (Implements pure virtual method in BaseSegment class).
   ///
   bool operator ==(const BaseSegment& other) const
   {
      const RangeStrideSegment* o_ptr = 
            dynamic_cast<const RangeStrideSegment*>(&other);
      if ( o_ptr ) {
        return ( *this == *o_ptr );
      } else {
        return false;
      }
   }

   ///
   /// Inquality operator returns true if segments are not equal; else false.
   /// (Implements pure virtual method in BaseSegment class).
   ///
   bool operator !=(const BaseSegment& other) const
   {
      return ( !(*this == other) );
   }

   ///
   /// Print segment data to given output stream.
   ///
   void print(std::ostream& os) const;

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
