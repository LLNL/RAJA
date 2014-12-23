/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining index set classes.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSet_HXX
#define RAJA_IndexSet_HXX


#include "RangeSegment.hxx"
#include "ListSegment.hxx"

#include "execpolicy.hxx"

#include "RAJAVec.hxx"

#if defined(RAJA_USE_STL)
#include <utility>
#endif

#include <iosfwd>


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Class representing an index set which is a collection
 *         of segment objects. 
 *
 ******************************************************************************
 */
class IndexSet
{
public:

   ///
   /// Nested class representing index set execution policy. 
   ///
   /// The first template parameter describes the policy for iterating
   /// over segments.  The second describes the execution policy for 
   /// each segment.
   ///
   template< typename SEG_ITER_POLICY_T,
             typename SEG_EXEC_POLICY_T > struct ExecPolicy
   {
      typedef SEG_ITER_POLICY_T seg_it;
      typedef SEG_EXEC_POLICY_T seg_exec;
   };

   ///
   /// Sequential execution policy for index set.
   ///
   typedef ExecPolicy<RAJA::seq_segit, RAJA::seq_exec> seq_policy;

   ///
   /// Construct empty index set
   ///
   IndexSet();

   ///
   /// Copy-constructor for index set
   ///
   IndexSet(const IndexSet& other);

   ///
   /// Copy-assignment operator for index set
   ///
   IndexSet& operator=(const IndexSet& rhs);

   ///
   /// Destroy index set including all index set segments.
   ///
   ~IndexSet();

   ///
   /// Swap function for copy-and-swap idiom.
   ///
   void swap(IndexSet& other);

   ///
   /// Add RangeSegment to back end of index set.
   ///
   void push_back(const RangeSegment& segment);

   ///
   /// Add RangeSegment to front end of index set.
   ///
   void push_front(const RangeSegment& segment);

#if 0  // RDH RETHINK
   ///
   /// Add RangeStrideSegment to back end of index set.
   ///
   void push_back(const RangeStrideSegment& segment);

   ///
   /// Add RangeStrideSegment to front end of index set.
   ///
   void push_front(const RangeStrideSegment& segment);
#endif

   ///
   /// Add ListSegment to back end of index set.
   ///
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned'  
   /// is passed to method, the new segment object does not own its indices
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   ///
   void push_back(const ListSegment& segment, 
                  IndexOwnership indx_own = Owned);

   ///
   /// Add ListSegment to front end of index set.
   ///
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned'
   /// is passed to method, the new segment object does not own its indices
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   ///
   void push_front(const ListSegment& segment,
                   IndexOwnership indx_own = Owned);

   ///
   /// Return total length of index set; i.e., sum of lengths
   /// of all segments.
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Return total number of segments in index set.
   ///
   int getNumSegments() const { 
      return m_segments.size(); 
   } 

   ///
   /// Return const pointer to BaseSegment object for segment 'i'.
   /// 
   /// Notes: No error-checking on segment index.
   ///
   ///        Object must be explicitly cast to proper type to
   ///        access actual segment index information 
   ///        (see BaseSegment::getType() method).
   ///
   const BaseSegment* getSegment(int i) const { 
      return m_segments[i]; 
   }

   ///
   /// Return non-const pointer to BaseSegment object for segment 'i'.
   ///
   /// Notes: No error-checking on segment index.
   ///
   ///        Object must be explicitly cast to proper type to
   ///        access actual segment index information 
   ///        (see BaseSegment::getType() method).
   ///
   BaseSegment* getSegment(int i) {
      return m_segments[i];       
   }

   ///
   /// Print index set data, including segments, to given output stream.
   ///
   void print(std::ostream& os) const;

private:
   //
   // Copy function for copy-and-swap idiom (deep copy).
   //
   void copy(const IndexSet& other);

   ///
   /// Helper function to add segment to back end of index set.
   ///
   template< typename SEG_T> 
   void push_back_private(SEG_T* seg)
   {
      seg->setIcount( m_len );
      m_segments.push_back( seg );

      m_len += seg->getLength();
   } 

   ///
   /// Helper function to add segment to front end of index set.
   ///
   template< typename SEG_T>
   void push_front_private(SEG_T* seg)
   {
      seg->setIcount( 0 );
      m_segments.push_front( seg );
      m_len += seg->getLength();

      Index_type icount = seg->getLength(); 
      for (unsigned i = 1; i < m_segments.size(); ++i ) {

         BaseSegment* iseg = getSegment(i);
         iseg->setIcount(icount); 
       
         SegmentType segtype = iseg->getType();
         
         switch ( segtype ) {

            case _RangeSeg_ : {
               icount += static_cast<RangeSegment*>(iseg)->getLength();
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStrideSeg_ : {
               icount += static_cast<RangeStrideSegment*>(iseg)->getLength();
               break;
            }
#endif

            case _ListSeg_ : {
               icount += static_cast<ListSegment*>(iseg)->getLength();
               break;
            }

            default : {
            }

         }  // switch ( segtype )
      }

   }

   ///
   Index_type  m_len;
   RAJAVec<BaseSegment*> m_segments;

}; 


}  // closing brace for RAJA namespace


#if defined(RAJA_USE_STL)
/*!
 ******************************************************************************
 *
 *  \brief Specialization of std swap method.
 *
 ******************************************************************************
 */
namespace std {

template< > 
RAJA_INLINE
void swap(RAJA::IndexSet& a, RAJA::IndexSet& b)
{
   a.swap(b);
}

}
#endif

#endif  // closing endif for header file include guard
