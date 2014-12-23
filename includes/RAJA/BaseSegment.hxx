/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining segment base class.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_BaseSegment_HXX
#define RAJA_BaseSegment_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

namespace RAJA {

/*!
 ******************************************************************************
 *
 * \brief  Base class for all segment classes.
 *
 ******************************************************************************
 */
class BaseSegment
{
public:

   ///
   /// Default ctor for base segment type.
   ///
   BaseSegment(SegmentType type, Index_type len)
      : m_type(type), m_len(len), m_icount(0) { ; }

   /*
    * Using compiler-generated copy ctor, copy assignment.
    */

   ///
   /// Virtual dtor.
   ///
   virtual ~BaseSegment() { ; }

   ///
   /// Get index count associated with start of segment.
   ///
   SegmentType getType() const { return m_type; }

   ///
   /// Get segment length (i.e., number of indices in segment).
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Set index count associated with start of segment.
   /// This is typically used when segment is part of an index set.
   ///
   void setIcount(Index_type icount) { m_icount = icount; }

   ///
   /// Get index count associated with start of segment.
   ///
   Index_type getIcount() const { return m_icount; }

   
   //
   // Pure virtual methods that must be provided by concrete segment classes.
   //

   ///
   /// Return enum value indicating whether segment owns the data rapresenting
   /// its indices.
   ///
   virtual IndexOwnership getIndexOwnership() const = 0;


private:
   //
   // The default ctor is not implemented.
   //
   BaseSegment();

   SegmentType m_type;

   Index_type m_len;
   Index_type m_icount;
}; 


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
