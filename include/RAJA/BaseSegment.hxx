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

#include <iosfwd>


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
   /// Ctor for base segment type.
   ///
   explicit BaseSegment(SegmentType type)
      : m_type(type), m_private(0) { ; }

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
   /// Retrieve pointer to private data. Must be cast to proper type by user.
   ///
   void* getPrivate() const { return m_private ; }

   ///
   /// Set pointer to private data. Can be used to associate any data 
   /// to segment. 
   ///
   /// NOTE: Caller retains ownership of data object.
   ///
   void setPrivate(void *ptr) { m_private = ptr ; }

   //
   // Pure virtual methods that must be provided by concrete segment classes.
   //

   ///
   /// Get segment length (i.e., number of indices in segment).
   ///
   virtual Index_type getLength() const = 0;

   ///
   /// Return enum value indicating whether segment owns the data rapresenting
   /// its indices.
   ///
   virtual IndexOwnership getIndexOwnership() const = 0;

   ///
   /// Pure virtual equality operator returns true if segments are equal; 
   /// else false.
   ///
   virtual bool operator ==(const BaseSegment& other) const = 0;

   ///
   /// Pure virtual inequality operator returns true if segments are not 
   /// equal, else false.
   ///
   virtual bool operator !=(const BaseSegment& other) const = 0;

private:
   ///
   /// The default ctor is not implemented.
   ///
   BaseSegment();

   ///
   /// Enum value indicating segment type.
   /// 
   SegmentType m_type;

   ///
   /// Pointer that can be used to hold arbitrary data associated with segment. 
   /// 
   void*       m_private;
}; 


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
