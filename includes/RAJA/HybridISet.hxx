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

#include "int_datatypes.hxx"

#include "execpolicy.hxx"

#include "RAJAVec.hxx"

#if defined(RAJA_USE_STL)
#include <utility>
#endif

#include <iosfwd>


namespace RAJA {

class RangeISet;
class UnstructuredISet;


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
   /// Nested class representing hybrid index set execution policy. 
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
   /// Sequential execution policy for hybrid index set.
   ///
   typedef ExecPolicy<RAJA::seq_segit, RAJA::seq_exec> seq_policy;

   ///
   /// Construct empty hybrid index set
   ///
   HybridISet();

   ///
   /// Construct hybrid index set from given index array using parameterized
   /// method buildHybridISet().
   ///
   HybridISet(const Index_type* const indices_in, Index_type length);

#if defined(RAJA_USE_STL)
   ///
   /// Construct hybrid index set from arbitrary object containing indices
   /// using parametrized method buildHybridISet().
   ///
   /// The object must provide the methods: size(), begin(), end().
   ///
   template< typename T> explicit HybridISet(const T& indx);
#endif

   ///
   /// Copy-constructor for hybrid index set
   ///
   HybridISet(const HybridISet& other);

   ///
   /// Copy-assignment operator for hybrid index set
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
   /// Add contiguous index range segment to hybrid index set 
   /// (adds RangeISet object).
   /// 
   void addRangeIndices(Index_type begin, Index_type end);

   ///
   /// Add RangeISet segment to hybrid index set.
   ///
   void addISet(const RangeISet& iset);

#if 0  // RDH RETHINK
   ///
   /// Add contiguous range of indices with stride segment to hybrid index set 
   /// (addds RangeStrideISet object).
   /// 
   void addRangeStrideIndices(Index_type begin, Index_type end, 
                              Index_type stride);

   ///
   /// Add RangeStrideISet segment to hybrid index set.
   ///
   void addISet(const RangeStrideISet& iset);
#endif

   ///
   /// Add segment containing array of indices to hybrid index set 
   /// (adds UnstructuredISet object).
   /// 
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned' 
   /// is passed to method, the new segment object does not own its indices 
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   /// 
   void addUnstructuredIndices(const Index_type* indx, Index_type len,
                               IndexOwnership indx_own = Owned);

   ///
   /// Add UnstructuredISet segment to hybrid index set.
   /// By default, the method makes a deep copy of given array and index
   /// set object will own the data representing its indices.  If 'Unowned'  
   /// is passed to method, the new segment object does not own its indices
   /// (i.e., it holds a handle to given array).  In this case, caller is
   /// responsible for managing object lifetimes properly.
   ///
   void addISet(const UnstructuredISet& iset, 
                IndexOwnership indx_own = Owned);

   ///
   /// Return total length of hybrid index set; i.e., sum of lengths
   /// of all segments.
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Return total number of segments in hybrid index set.
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
   const void* getSegmentISet(int i) const { 
      return m_segments[i].m_iset; 
   } 

   ///
   /// Return enum value indicating whether segment 'i' index set owns the 
   /// data representing its indices.
   /// 
   /// Note: No error-checking on segment index.
   ///
   IndexOwnership segmentIndexOwnership(int i) const {
      return m_segments[i].m_indx_own; 
   } 

   ///
   /// Print hybrid index set data, including segments, to given output stream.
   ///
   void print(std::ostream& os) const;

private:
   //
   // Copy function for copy-and-swap idiom (deep copy).
   //
   void copy(const HybridISet& other);

   ///
   /// Private nested class to hold an index segment of a hybrid index set.
   ///
   /// A segment is defined by its type and its index set object.
   ///
   class Segment
   {
   public:
      Segment() 
         : m_type(_Unknown_), m_iset(0), m_indx_own(Unowned) { ; } 

      template <typename ISET>
      Segment(SegmentType type,  const ISET* iset)
         : m_type(type), m_iset(iset), m_indx_own(iset->indexOwnership()) { ; }

      ///
      /// Using compiler-provided dtor, copy ctor, copy-assignment.
      ///

      SegmentType m_type;
      const void* m_iset;
      IndexOwnership m_indx_own;
   };

   //
   // Helper function to add segment.
   //
   template< typename SEG_T> 
   void addSegment(SegmentType seg_type, const SEG_T* seg)
   {
      m_segments.push_back(Segment( seg_type, seg ));
      m_len += seg->getLength();
   } 

   ///
   Index_type  m_len;
   RAJAVec<Segment> m_segments;

}; 


/*!
 ******************************************************************************
 *
 * \brief Initialize hybrid index set from array of indices with given length.
 *
 *        Note given hybrid index set object is assumed to be empty.  
 *
 *        Routine does no error-checking on argements and assumes Index_type
 *        array contains valid indices.
 *
 ******************************************************************************
 */
void buildHybridISet(HybridISet& hiset,
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
HybridISet::HybridISet(const T& indx)
: m_len(0)
{
   std::vector<Index_type> vec(indx.begin(), indx.end());
   buildHybridISet(*this, &vec[0], vec.size());
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
void swap(RAJA::HybridISet& a, RAJA::HybridISet& b)
{
   a.swap(b);
}

}
#endif

#endif  // closing endif for header file include guard
