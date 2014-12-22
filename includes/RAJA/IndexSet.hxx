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

#include "config.hxx"

#include "int_datatypes.hxx"

#include "execpolicy.hxx"

#include "RAJAVec.hxx"

#if defined(RAJA_USE_STL)
#include <utility>
#endif

#include <iosfwd>


namespace RAJA {

class RangeSegment;
class ListSegment;


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
   /// Construct index set from given index array using parameterized
   /// method buildIndexSet().
   ///
   IndexSet(const Index_type* const indices_in, Index_type length);

#if defined(RAJA_USE_STL)
   ///
   /// Construct index set from arbitrary object containing indices
   /// using parametrized method buildIndexSet().
   ///
   /// The object must provide the methods: size(), begin(), end().
   ///
   template< typename T> explicit IndexSet(const T& indx);
#endif

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
   /// Append contiguous index range segment to back end of index set 
   /// (adds RangeSegment object).
   /// 
   void push_back_RangeSegment(Index_type begin, Index_type end);

   ///
   /// Add RangeSegment to back end of index set.
   ///
   void push_back_Segment(const RangeSegment& iset);

   ///
   /// Append contiguous index range segment to front end of index set
   /// (adds RangeSegment object).
   ///
   void push_front_RangeSegment(Index_type begin, Index_type end);

   ///
   /// Add RangeSegment to front end of index set.
   ///
   void push_front_Segment(const RangeSegment& iset);

#if 0  // RDH RETHINK
   ///
   /// Add contiguous range of indices with stride segment to back end 
   /// of index set (addds RangeStrideSegment object).
   /// 
   void push_back_RangeStrideSegment(Index_type begin, Index_type end, 
                                     Index_type stride);

   ///
   /// Add RangeStrideSegment to back end of index set.
   ///
   void push_back_Segment(const RangeStrideSegment& iset);

   ///
   /// Add contiguous range of indices with stride segment to front end 
   /// of index set (addds RangeStrideSegment object).
   /// 
   void push_front_RangeStrideSegment(Index_type begin, Index_type end, 
                                      Index_type stride);

   ///
   /// Add RangeStrideSegment to front end of index set.
   ///
   void push_front_Segment(const RangeStrideSegment& iset);
#endif

   ///
   /// Add segment containing array of indices to back end of index set 
   /// (adds ListSegment object).
   /// 
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned' 
   /// is passed to method, the new segment object does not own its indices 
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   /// 
   void push_back_ListSegment(const Index_type* indx, Index_type len,
                              IndexOwnership indx_own = Owned);

   ///
   /// Add ListSegment to back end of index set.
   ///
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned'  
   /// is passed to method, the new segment object does not own its indices
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   ///
   void push_back_Segment(const ListSegment& iset, 
                          IndexOwnership indx_own = Owned);

   ///
   /// Add segment containing array of indices to front end of index set
   /// (adds ListSegment object).
   ///
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned'
   /// is passed to method, the new segment object does not own its indices
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   ///
   void push_front_ListSegment(const Index_type* indx, Index_type len,
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
   void push_front_Segment(const ListSegment& iset,
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
   /// Return enum value defining type of segment 'i'.
   /// 
   /// Note: No error-checking on segment index.
   ///
   SegmentType getSegmentType(int i) const { 
      return m_segments[i].m_type; 
   }

   ///
   /// Return const void pointer to index set for segment 'i'.
   /// 
   /// Notes: Pointer must be explicitly cast to proper type before use
   ///        (see getSegmentType() method).
   ///
   ///        No error-checking on segment index.
   ///
   const void* getSegment(int i) const { 
      return m_segments[i].m_segment; 
   } 

   ///
   /// Return Index_type value indicating index count associated with
   /// start of segment 'i'.
   ///
   /// Note: No error-checking on segment index.
   ///
   Index_type getSegmentIcount(int i) const {
      return m_segments[i].m_icount;
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
   /// Private nested class to hold a segment of a index set.
   ///
   /// A segment is defined by its type and its index set object.
   ///
   /// The index count value can be provided as a second argument to 
   /// forall_Ioff( ) iteration methods to map between actual indices
   /// indices and the running iteration count. That is, the count 
   /// for a segment starts with the total length of all segments
   /// preceding that segment.
   ///
   class Segment
   {
   public:
      Segment() 
         : m_type(_UnknownSeg_), m_segment(0), m_icount(0) { ; } 

      template <typename SEG_T>
      Segment(SegmentType type,  const SEG_T* segment, Index_type icount)
         : m_type(type), m_segment(segment), m_icount(icount) { ; }

      ///
      /// Using compiler-provided dtor, copy ctor, copy-assignment.
      ///

      SegmentType m_type;
      const void* m_segment;

      Index_type m_icount;
   };

   ///
   /// Helper function to add segment to back end of index set.
   ///
   template< typename SEG_T> 
   void push_back_Segment_private(SegmentType seg_type, const SEG_T* seg)
   {
      m_segments.push_back(Segment( seg_type, seg, m_len ));
      m_len += seg->getLength();
   } 

   ///
   /// Helper function to add segment to front end of index set.
   ///
   template< typename SEG_T>
   void push_front_Segment_private(SegmentType seg_type, const SEG_T* seg)
   {
      m_segments.push_front(Segment( seg_type, seg, 0 ));
      m_len += seg->getLength();

      Index_type icount = seg->getLength(); 
      for (unsigned i = 1; i < m_segments.size(); ++i ) {
         m_segments[i].m_icount = icount;
        
         SegmentType segtype = getSegmentType(i);
         const void* iset = getSegment(i); 
         
         switch ( segtype ) {

            case _RangeSeg_ : {
               RangeSegment* is =
                  const_cast<RangeSegment*>(
                     static_cast<const RangeSegment*>(iset)
                  );
               icount += is->getLength();
               break;
            }

#if 0  // RDH RETHINK
            case _RangeStrideSeg_ : {
               RangeStrideSegment* is =
                  const_cast<RangeStrideSegment*>(
                     static_cast<const RangeStrideSegment*>(iset)
                  );
               icount += is->getLength();
               break;
            }
#endif

            case _ListSeg_ : {
               ListSegment* is =
                  const_cast<ListSegment*>(
                     static_cast<const ListSegment*>(iset)
                  );
               icount += is->getLength();
               break;
            }

            default : {
            }

         }  // switch ( segtype )
      }
   }

   ///
   Index_type  m_len;
   RAJAVec<Segment> m_segments;

}; 


/*!
 ******************************************************************************
 *
 * \brief Initialize index set from array of indices with given length.
 *
 *        Note given index set object is assumed to be empty.  
 *
 *        Routine does no error-checking on argements and assumes Index_type
 *        array contains valid indices.
 *
 ******************************************************************************
 */
void buildIndexSet(IndexSet& hiset,
                   const Index_type* const indices_in,
                   Index_type length);

#if defined(RAJA_USE_STL)
/*!
 ******************************************************************************
 *
 * \brief Implementation of generic constructor template.
 *
 ******************************************************************************
 */
template <typename T>
IndexSet::IndexSet(const T& indx)
: m_len(0)
{
   std::vector<Index_type> vec(indx.begin(), indx.end());
   buildIndexSet(*this, &vec[0], vec.size());
}
#endif


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
