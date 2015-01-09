/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining segment info class for index sets.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSetSegInfo_HXX
#define RAJA_IndexSetSegInfo_HXX

#include "int_datatypes.hxx"

namespace RAJA {

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
        m_icount(UndefinedValue) { ; }

   ///
   /// Ctor to create segment info for give segment.
   ///
#if 0
   template <typename SEG_T>
   IndexSetSegInfo(SEG_T* segment, bool owns_segment)
      : m_segment(segment),
        m_owns_segment(owns_segment),
        m_icount(UndefinedValue) { ; }
#else
   IndexSetSegInfo(BaseSegment* segment, bool owns_segment)
      : m_segment(segment),
        m_owns_segment(owns_segment),
        m_icount(UndefinedValue) { ; }
#endif

   ~IndexSetSegInfo() { ; }

   /*
    * Using compiler-provided copy ctor and copy-assignment.
    */

   ///
   /// Return const pointer to actual segment object.
   ///
   const BaseSegment* getSegment() const { return m_segment; }

   ///
   /// Return pointer to actual segment object.
   ///
   BaseSegment* getSegment() { return m_segment; }

   ///
   /// Get index count for start of segment.
   ///
   Index_type getIcount() const { return m_icount; }

   ///
   /// Set index count for start of segment.
   ///
   void setIcount(Index_type icount) { m_icount = icount; }

   ///
   /// Get index count for start of segment.
   ///
   bool ownsSegment() const { return m_owns_segment; }

private:
   BaseSegment* m_segment;
   bool         m_owns_segment;

   Index_type   m_icount;

}; 


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
