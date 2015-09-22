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

#include "IndexSetSegInfo.hxx"

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

//
// RDH TO DO: Add COMPILE TIME segment type selection.
//

   ///
   /// Nested class representing index set execution policy. 
   ///
   /// The first template parameter describes the policy for iterating
   /// over segments.  The second describes the policy for executing
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


//@{
//!  @name Constructor and destructor methods

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

//@}


//@{
//!  @name Segment insertion and accessor methods

   ///
   /// Return true if given segment is valid for this IndexSet class; 
   /// otherwise, return false.
   ///
   bool isValidSegmentType(const BaseSegment* segment) const;

   /*
    * IMPORTANT: Some methods to add a segment to an index set
    *            make a copy of the segment object passed in. Others do not.
    *
    *            The no-copy method names indicate the choice.
    *            The copy/no-copy methods are further distinguished 
    *            by taking a const reference (copy) or non-const 
    *            pointer (no-copy).
    *
    *            Each method returns true if segment is added successfully;
    *            false otherwise.
    */

   ///
   /// Add segment to back end of index set without making a copy.
   ///
   bool push_back_nocopy(BaseSegment* segment) 
   { return( push_back_private(segment, false /* does not own segment */) ); } 

   ///
   /// Add segment to front end of index set without making a copy.
   ///
   bool push_front_nocopy(BaseSegment* segment)
   { return( push_front_private(segment, false /* does not own segment */) ); }

   ///
   /// Add copy of segment to back end of index set.
   ///
   bool push_back(const BaseSegment& segment);

   ///
   /// Add copy of segment to front end of index set.
   ///
   bool push_front(const BaseSegment& segment);


   ///
   /// Return total length of index set; i.e., sum of lengths
   /// of all segments.
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Return total number of segments in index set.
   ///
   unsigned getNumSegments() const { 
      return m_segments.size(); 
   }

   ///
   /// Return const pointer to BaseSegment 'i'.
   ///
   /// Notes: No error-checking on segment index.
   ///
   ///        Object must be explicitly cast to concrete type to
   ///        access actual segment index information
   ///        (see BaseSegment::getType() method).
   ///
   const BaseSegment* getSegment(unsigned i) const {
      return m_segments[i].getSegment();
   }

   ///
   /// Return non-const pointer to BaseSegment 'i'.
   ///
   /// Notes: No error-checking on segment index.
   ///
   ///        Object must be explicitly cast to concrete type to
   ///        access actual segment index information
   ///        (see BaseSegment::getType() method).
   ///
   BaseSegment* getSegment(unsigned i) {
      return m_segments[i].getSegment();
   } 

   ///
   /// Return const pointer to IndexSetSegInfo object for segment 'i'.
   /// 
   /// Note: No error-checking on segment index.
   ///
   const IndexSetSegInfo* getSegmentInfo(unsigned i) const { 
      return &(m_segments[i]); 
   }

   ///
   /// Return non-const pointer to BaseSegment object for segment 'i'.
   ///
   /// Note: No error-checking on segment index.
   ///
   IndexSetSegInfo* getSegmentInfo(unsigned i) {
      return &(m_segments[i]);
   }

//@}


//@{
//!  @name IndexSet segment subsetting methods (views ranges)

   ///
   /// Return a new IndexSet object that contains the subset of
   /// segments in this IndexSet with ids in the interval [begin, end).
   ///
   /// This IndexSet will not change and the created "view" into it 
   /// will not own any of its segments.
   ///
   IndexSet* createView(int begin, int end) const;

   ///
   /// Return a new IndexSet object that contains the subset of
   /// segments in this IndexSet with ids in the given int array.
   ///
   /// This IndexSet will not change and the created "view" into it 
   /// will not own any of its segments.
   ///
   IndexSet* createView(const int* segIds, int len) const;

   ///
   /// Return a new IndexSet object that contains the subset of
   /// segments in this IndexSet with ids in the argument object.
   ///
   /// This IndexSet will not change and the created "view" into it 
   /// will not own any of its segments.
   ///
   /// The object must provide methods begin(), end(), and its
   /// iterator type must de-reference to an integral value.
   ///
   template< typename T> 
   IndexSet* createView(const T& segIds) const;

   
   ///
   /// Set [begin, end) interval of segment ids identified by
   /// given interval id.
   ///
   /// For example, this method can be used to assign an interval of 
   /// segments to be processed by a given thread.
   ///
   void setSegmentInterval(int interval_id, int begin, int end);

   ///
   /// Get lower bound or upper bound of [begin, end) interval of segment 
   /// ids identified by given interval id.
   ///
   /// Notes: No error-checking on interval id.
   ///
   int getSegmentIntervalBegin(int interval_id) const {
      return m_seg_interval_begin[interval_id];
   }
   ///
   int getSegmentIntervalEnd(int interval_id) const {
      return m_seg_interval_end[interval_id];
   }
   

//@}


//@{
//!  @name Private data set/get methods

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

//@}


//@{
//!  @name Segment dependency methods

   ///
   /// Return true if dependencyGraphFinalize() method has been called 
   /// on index set object. 
   ///
   bool dependencyGraphSet() const { return m_dep_graph_set; }

   ///
   /// Create dependency graph node objects (one for each segment in index 
   /// set and initialize each to default state.
   ///
   /// Note that dependency graph data for segments needs to be set for
   /// each dependency graph node for dependency-graph scheduling to work
   /// properly. See DepGraphNode class.  After setting all dependency
   /// graph data, the dependencyGraphFinalize() method should be called
   /// to indicate that dependency graph is complete.
   ///
   /// Note that this method assumes dependency graph node objects don't 
   /// already exist for index set.
   ///
   void initDependencyGraph();

   ///
   /// Calling this method indicates that dependency graph data for all
   /// segments in index set has been set.
   ///
   /// This method should be called after all such data has been set.
   ///
   void dependencyGraphFinalize() { m_dep_graph_set = true; }

//@}

//@{
//!  @name Index set equality/inequality check methods

   ///
   /// Equality operator returns true if all segments are equal; else false.
   /// 
   /// Note: method does not check equality of anything other than segment 
   ///       types and indices; e.g., dependency info not checked.
   ///
   bool operator ==(const IndexSet& other) const ;

   ///
   /// Inequality operator returns true if any segment is not equal, else false.
   ///
   bool operator !=(const IndexSet& other) const
   {
      return ( !(*this == other) );
   }

//@}


   ///
   /// Print index set data, including segments, to given output stream.
   ///
   void print(std::ostream& os) const;

private:
   ///
   /// Copy function for copy-and-swap idiom (deep copy).
   ///
   void copy(const IndexSet& other);

   ///
   /// Helper function to add segment to back end of index set.
   /// Returns true if segment added, false otherwise.
   ///
   bool push_back_private(BaseSegment* seg, bool owns_segment);

   ///
   /// Helper function to add segment to front end of index set.
   /// Returns true if segment added, false otherwise.
   ///
   bool push_front_private(BaseSegment* seg, bool owns_segment);

   ///
   /// Helper function to create a copy of a given segment given a 
   /// pointer to the BaseSegment.
   ///
   BaseSegment* createSegmentCopy(const BaseSegment& segment) const;



   ///
   /// Total length of all IndexSet segments.
   ///
   Index_type  m_len;

   ///
   /// Collection of IndexSet segment info objects.
   ///
   RAJAVec<IndexSetSegInfo> m_segments;

   ///
   /// Vectors holding user defined segment intervals; each is [begin, end).
   ///
   RAJAVec<int> m_seg_interval_begin;
   RAJAVec<int> m_seg_interval_end;

   ///
   /// Pointer for holding arbitrary data associated with index set.
   ///
   void*       m_private;

   ///
   ///  True if dependencyGraphFinalize() method has been called; 
   ///  else false (default).
   ///
   bool m_dep_graph_set;

}; 

/*!
 ******************************************************************************
 *
 *  \brief Implementation of generic IndexSet "view" template.
 *
 ******************************************************************************
 */
template< typename T>
IndexSet* IndexSet::createView(const T& segIds) const
{
   IndexSet *retVal = new IndexSet() ;

   int numSeg = m_segments.size() ;
   for (auto it = segIds.begin(); it != segIds.end(); ++it) {
      if (*it >= 0 && *it < numSeg) {
         retVal->push_back_nocopy( 
            const_cast<BaseSegment*>( m_segments[ *it ].getSegment() ) ) ;
      }
   }

   return retVal ;
}


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
