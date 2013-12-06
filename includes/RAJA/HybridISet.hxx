/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining hybrid index set classes.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_HybridISet_HXX
#define RAJA_HybridISet_HXX

#include "config.hxx"

#include "datatypes.hxx"

#include "execpolicy.hxx"

#include "RangeISet.hxx"
#include "UnstructuredISet.hxx"

#include <vector>
#include <utility>
#include <iostream>


namespace RAJA {

/*!
 ******************************************************************************
 *
 * \brief  Class representing an hybrid index set which is a collection
 *         of index set objects defined above.  Within a hybrid, the
 *         individual index sets are referred to as segments.
 *
 ******************************************************************************
 */
class HybridISet
{
public:

   ///
   /// Sequential execution policy for hybrid index set.
   ///
   typedef std::pair<RAJA::seq_segit, RAJA::seq_exec> seq_policy;

   ///
   /// Enum describing types of segments in hybrid index set.
   ///
   enum SegmentType { _Range_, _RangeStride_, _Unstructured_, _Unknown_ };

   ///
   /// Class holding a segment and its segment type.
   ///
   class Segment
   {
   public:

      Segment() 
         : m_type(_Unknown_), m_segment(0) { ; } 

      Segment(SegmentType type,  const void* segment) 
         : m_type(type), m_segment(segment) { ; }

      SegmentType m_type;
      const void* m_segment;
       
   };

   ///
   /// Construct empty hybrid index set
   ///
   HybridISet();

   ///
   /// Copy-constructor for hybrid index set
   ///
   HybridISet(const HybridISet& other);

   ///
   /// Copy-assignment for hybrid index set
   ///
   HybridISet& operator=(const HybridISet& rhs);

   ///
   /// Destroy index set including all index set segments.
   ///
   ~HybridISet();

   ///
   /// Swap function for copy-and-swap idiom.
   ///
   void swap(HybridISet& other);

   ///
   /// Create copy of given index set object and add to hybrid index set.
   ///
   template< typename INDEXSET_T >
   void addISet(const INDEXSET_T& index_set);

   ///
   /// Add contiguous range of indices to hybrid index set as a RangeISet.
   /// 
   void addRangeIndices(Index_type begin, Index_type end);

#if 0  // RDH RETHINK
   ///
   /// Add contiguous range of indices with stride to hybrid index set 
   /// as a RangeStrideISet.
   /// 
   void addRangeStrideIndices(Index_type begin, Index_type end, Index_type stride);
#endif

   ///
   /// Add array of indices to hybrid index set as an UnstructuredISet. 
   /// 
   void addUnstructuredIndices(const Index_type* indx, Index_type len);

   ///
   /// Return total length of hybrid index set; i.e., sum of lengths
   /// over all segments.
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Return total number of segments in hybrid index set.
   ///
   int getNumSegments() const { return m_segments.size(); } 

   ///
   /// Return const reference to segment 'i' in hybrid index set.
   ///
   const Segment& getSegment(int i) const { return m_segments[i]; } 

   ///
   /// Return const pointer to array of segments in hybrid index set.
   ///
   const Segment* getSegments() const { return &m_segments[0]; }

   void print(std::ostream& os) const;

private:
   //
   // Copy function for copy-and-swap idiom (deep copy).
   //
   void copy(const HybridISet& other);

   Index_type  m_len;
   std::vector<Segment> m_segments;

}; 


/*!
 ******************************************************************************
 *
 * \brief Return pointer to hybrid index set created from array of indices 
 *        with given length.
 *
 *        Routine does no error-checking on argements and assumes Index_type
 *        array contains valid (non-negative) indices.
 *
 *        Caller assumes ownership of returned index set.
 *
 ******************************************************************************
 */
HybridISet* buildHybridISet(const Index_type* const indices_in,
                            Index_type length);

/*!
 ******************************************************************************
 *
 * \brief Same as above, but takes std::vector of indices.
 *
 ******************************************************************************
 */
RAJA_INLINE
HybridISet* buildHybridISet(const std::vector<Index_type>& indices)
{
   if ( indices.size() > 0 ) {
      return( buildHybridISet(&indices[0], indices.size()) );
   } else {
      HybridISet* hindex = new HybridISet();
      return( hindex );
   }
}


}  // closing brace for namespace statement

#endif  // closing endif for header file include guard
