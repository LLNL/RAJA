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

#include "DepGraphNode.hxx"


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
 *         The dependency graph node member can be used to define a 
 *         dependency-graph among segments in an index set.  By default it
 *         is not used.
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
        m_icount(UndefinedValue), 
        m_dep_graph_node(0) { ; }

   ///
   /// Ctor to create segment info for give segment.
   ///
   IndexSetSegInfo(BaseSegment* segment, bool owns_segment)
      : m_segment(segment),
        m_owns_segment(owns_segment),
        m_icount(UndefinedValue), 
        m_dep_graph_node(0) { ; }

   ~IndexSetSegInfo() { if (m_dep_graph_node) delete m_dep_graph_node; }

   /*
    * Using compiler-provided copy ctor and copy-assignment.
    */

   ///
   /// Retrieve const pointer to base-type segment object.
   ///
   const BaseSegment* getSegment() const { return m_segment; }

   ///
   /// Retrieve pointer to base-type segment object.
   ///
   BaseSegment* getSegment() { return m_segment; }

   ///
   /// Return true if IndexSetSegInfo object owns segment (set at construction);
   /// false, otherwise. False usually means segment is shared by some other
   /// IndexSetSegInfo that owns it. 
   ///
   bool ownsSegment() const { return m_owns_segment; }

   ///
   /// Set/get index count for start of segment.
   ///
   void setIcount(Index_type icount) { m_icount = icount; }
   ///
   Index_type getIcount() const { return m_icount; }

   ///
   /// Retrieve const pointer to dependency graph node object for segment.
   ///
   const DepGraphNode* getDepGraphNode() const { return m_dep_graph_node; }

   ///
   /// Retrieve pointer to dependency graph node object for segment.
   ///
   DepGraphNode* getDepGraphNode() { return m_dep_graph_node; }
  
   ///
   /// Create dependency graph node object for segment and 
   /// initialize to default state.
   ///
   /// Note that this method assumes dep node object doesn't already exist.
   ///
   void initDepGraphNode() { m_dep_graph_node = new DepGraphNode(); }
  

private:
   BaseSegment* m_segment;
   bool         m_owns_segment;

   Index_type   m_icount;

   DepGraphNode* m_dep_graph_node;

}; 


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
