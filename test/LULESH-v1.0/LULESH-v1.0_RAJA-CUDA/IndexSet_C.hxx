/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main Index Set API header file.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 *
 ******************************************************************************
 */

#ifndef HybridIndexSet_HXX
#define HybridIndexSet_HXX

#include <vector>
#include <iostream>
#include <cstdlib>

#include "RAJA/RAJA.hxx"

#define RAJA_SYNC 1

#define FT_BEGIN
#define FT_END

namespace RAJA {

#if 1 // RDH 
/*!
 ******************************************************************************
 *
 * \brief  Class representing a contiguous range of indices.
 *
 *         Range is specified by begin and end values.
 *         Traversal executes as:  
 *            for (i = m_begin; i < m_end; ++i) {
 *               expression using i as array index.
 *            }
 *
 ******************************************************************************
 */
class RangeIndexSet
{
public:

   RangeIndexSet(Index_type begin, Index_type end) 
     : m_begin(begin), m_end(end)
   { if (end-begin < RAJA::RANGE_MIN_LENGTH ) /* abort */ ; }

   Index_type getBegin() const { return m_begin; }
   Index_type getEnd() const { return m_end; }

   Index_type getLength() const { return (m_end-m_begin); }

private:
   //
   // The default ctor is not implemented.
   //
   RangeIndexSet();

   Index_type m_begin;
   Index_type m_end;
};
#endif

/*!
 ******************************************************************************
 *
 * \brief  Class representing an arbitrary collection of indices. 
 *
 *         Length indicates number of indices in index array.
 *         Traversal executes as:  
 *            for (i = 0; i < m_len; ++i) {
 *               expression using m_indx[i] as array index.
 *            }
 *
 ******************************************************************************
 */
class UnstructuredIndexSet
{
public:

   UnstructuredIndexSet(Index_type begin, Index_type end) ;

   UnstructuredIndexSet(const Index_type* indx, Index_type len);

   UnstructuredIndexSet(const UnstructuredIndexSet& obj);

   UnstructuredIndexSet& operator=(const UnstructuredIndexSet& other);

   ~UnstructuredIndexSet();

   Index_type getLength() const { return m_len; }

   const Index_type* getIndex() const { return m_indx; }

private:
   //
   // The default ctor is not implemented.
   //
   UnstructuredIndexSet();

   //
   // Allocate new index array, copy indices to it and return it.
   //
   Index_type* copyIndices(const Index_type* indx, Index_type len);

   Index_type* __restrict__ m_indx;
   Index_type  m_len;
};


/*!
 ******************************************************************************
 *
 * \brief  Class representing an hybrid index set comprised of a collection
 *         of contiguous ranges and arbitrary indices.  
 *
 *         Each element in collection is called a "part" of the hybrid 
 *         index set.
 *
 ******************************************************************************
 */
class HybridIndexSet
{
public:

   ///
   /// Enum describing types of parts in hybrid index set and typedef
   /// to hold a single part
   ///
   enum IndexType { _Range_, _Unstructured_ };
   ///
   typedef std::pair<IndexType, const void*> PartPair;

   ///
   /// Construct empty hybrid index set
   ///
   HybridIndexSet();

   //
   // Copy-constructor for hybrid index set
   //
   HybridIndexSet(const HybridIndexSet&);

   //
   // Copy-assignment for hybrid index set
   //
   HybridIndexSet& operator=(const HybridIndexSet&);

   ///
   /// Hybrid index set destructor destroys all index set parts.
   ///
   ~HybridIndexSet();

   ///
   /// Create copy of given RangeIndexSet and add to hybrid index set.
   ///
   void addIndexSet(const RangeIndexSet& index_set);

   ///
   /// Add contiguous range of indices to hybrid index set as a RangeIndexSet.
   /// 
   void addRangeIndices(Index_type begin, Index_type end);

   ///
   /// Create copy of given UnstructuredIndexSet and add to hybrid index set.
   ///
   void addIndexSet(const UnstructuredIndexSet& index_set);

   ///
   /// Add array of indices to hybrid index set as an UnstructuredIndexSet. 
   /// 
   void addUnstructuredIndices(const Index_type* indx, Index_type len);

   ///
   /// Return total length of hybrid index set; i.e., sum of lengths
   /// over all parts.
   ///
   Index_type getLength() const { return m_len; }

   ///
   /// Return total length of hybrid index set; i.e., sum of lengths
   /// over all parts.
   ///
   Index_type getNumParts() const { return m_indexsets.size(); } 

   ///
   /// Return PartPair at given part index in hybrid index set.
   ///
   const PartPair& getPartPair(int part_isi) const 
      { return m_indexsets[part_isi]; }

private:
   //
   // Copy parts (deep copy) to given vector and return total length.
   //
   void copyParts(const HybridIndexSet& other);

   Index_type  m_len;
   std::vector<PartPair> m_indexsets;

}; 


/*!
 ******************************************************************************
 *
 * \brief  Traverse HybridIndexSet using traversal method specified by
 *         template parameter on all parts.
 *
 ******************************************************************************
 */
template <typename TRAVERSE_T, typename LOOP_BODY>
void IndexSet_forall(TRAVERSE_T traversal,
            const HybridIndexSet& is, LOOP_BODY loop_body)
{
   for ( int isi = 0; isi < is.getNumParts(); ++isi ) {
      const HybridIndexSet::PartPair& is_pair = is.getPartPair(isi);

      switch ( is_pair.first ) {

         case HybridIndexSet::_Range_ : {
            const RangeIndexSet* iseg = 
               static_cast<const RangeIndexSet*>(is_pair.second);
            forall(traversal,
               iseg->getBegin(), iseg->getEnd(),
               loop_body
            );
            break;
         }

         case HybridIndexSet::_Unstructured_ : {
            const UnstructuredIndexSet* iseg = 
               static_cast<const UnstructuredIndexSet*>(is_pair.second);
            forall(traversal,
               iseg->getIndex(), iseg->getLength(),
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch ( is_pair.first ) 

   } // iterate over parts of hybrid index set
}

template <typename TRAVERSE_T, typename LOOP_BODY>
void IndexSet_forall_minloc(TRAVERSE_T traversal,
                   const HybridIndexSet& is,
                   double *min, int *loc,
                   LOOP_BODY loop_body)
{
   for ( int isi = 0; isi < is.getNumParts(); ++isi ) {
      const HybridIndexSet::PartPair& is_pair = is.getPartPair(isi);

      switch ( is_pair.first ) {

         case HybridIndexSet::_Range_ : {
            const RangeIndexSet* iseg =
               static_cast<const RangeIndexSet*>(is_pair.second);
            forall_minloc(traversal,
               iseg->getBegin(), iseg->getEnd(),
               min, loc, 
               loop_body
            );
            break;
         }

         case HybridIndexSet::_Unstructured_ : {
            const UnstructuredIndexSet* iseg =
               static_cast<const UnstructuredIndexSet*>(is_pair.second);
            forall_minloc(traversal,
               iseg->getIndex(), iseg->getLength(),
               min, loc, 
               loop_body
            );
            break;
         }

         default : {
         }
      }  // switch ( is_pair.first ) 
   } // iterate over parts of hybrid index set
}

/*!
 ******************************************************************************
 *
 * \brief Traversal of index set using template parameter to specify traversal. 
 *
 ******************************************************************************
 */
template <typename TRAVERSE_T,
          typename INDEXSET_T, typename LOOP_BODY>
RAJA_INLINE
void IndexSet_forall(const INDEXSET_T& is, LOOP_BODY loop_body)
{
   IndexSet_forall(TRAVERSE_T(), is, loop_body);
}

template <typename TRAVERSE_T,
          typename INDEXSET_T, typename LOOP_BODY>
RAJA_INLINE
void IndexSet_forall_minloc(const INDEXSET_T& is, double *min, int *loc,
                     LOOP_BODY loop_body)
{
   IndexSet_forall_minloc(TRAVERSE_T(), is, min, loc, loop_body);
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
