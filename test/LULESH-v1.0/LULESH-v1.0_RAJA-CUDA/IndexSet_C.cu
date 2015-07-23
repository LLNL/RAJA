/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for index set classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 *
 ******************************************************************************
 */

#include "IndexSet_C.hxx"

#include <cstdlib>
#include <iostream>

namespace RAJA {

#if 0 // RDH

/*
*************************************************************************
*
* UnstructuredIndexSet class methods
*
*************************************************************************
*/

UnstructuredIndexSet::UnstructuredIndexSet(Index_type begin, Index_type end)
{
   m_len = end - begin ;
   if ( m_len > 0 ) {
#if defined(RAJA_USE_CUDA)
    if (cudaMallocManaged((void **)&m_indx, m_len*sizeof(Index_type), cudaMemAttachGlobal) 
								!= cudaSuccess)
    {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
    }
#else
      posix_memalign((void **)&m_indx,
                     RAJA::DATA_ALIGN, m_len*sizeof(Index_type)) ;
		     
#endif
		     
      for (Index_type i = begin; i < end; ++i) {
         m_indx[i - begin] = i ;
      }
   } 
   else {
      m_len = 0 ;
      m_indx = 0 ;
   }
}

UnstructuredIndexSet::UnstructuredIndexSet(const Index_type* indx, Index_type len)
: m_indx(0), m_len(len)
{
   m_indx = copyIndices(indx, len);
}

UnstructuredIndexSet::UnstructuredIndexSet(const UnstructuredIndexSet& other)
: m_indx(0), m_len(other.m_len)
{
   m_indx = copyIndices(other.m_indx, m_len);
}

UnstructuredIndexSet& UnstructuredIndexSet::operator=(
   const UnstructuredIndexSet& rhs)
{
   if (this != &rhs) {
      m_len = rhs.m_len;
      m_indx = copyIndices(rhs.m_indx, m_len);
   }
   return *this;
}

UnstructuredIndexSet::~UnstructuredIndexSet()
{
#if defined(RAJA_USE_CUDA)
  if (m_indx) {
    if (cudaFree(m_indx) != cudaSuccess) {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
    }
  }
#else
   if (m_indx) free( m_indx );
#endif   
}


/*
*************************************************************************
*
* Private helper function to copy array of indices.
*
*************************************************************************
*/
Index_type* UnstructuredIndexSet::copyIndices(const Index_type* indx, 
                                             Index_type len)
{
   Index_type* tindx = 0;
   if ( len > 0 ) {
#if defined(RAJA_USE_CUDA)
      if (cudaMallocManaged((void **)&tindx, len*sizeof(Index_type), cudaMemAttachGlobal)
						           != cudaSuccess) 
      {
	std::cerr << "\n ERROR line " << __LINE__ 
		  << " cudaMallocManaged failed!" << std::endl;
	exit(1);
      }	
#else
      posix_memalign((void **)&tindx,
                     RAJA::DATA_ALIGN, len*sizeof(Index_type)) ;
#endif		     
      for (Index_type i = 0; i < len; ++i) {
         tindx[i] = indx[i] ;
      }
   } 
   return tindx;
}


/*
*************************************************************************
*
* HybridIndexSet class constructor and destructor.
*
*************************************************************************
*/

HybridIndexSet::HybridIndexSet()
: m_len(0)
{
}

HybridIndexSet::HybridIndexSet(const HybridIndexSet& other)
: m_len(0)
{
   copyParts(other); 
}

HybridIndexSet& HybridIndexSet::operator=(
   const HybridIndexSet& rhs)
{
   if (this != &rhs) {
      copyParts(rhs);
   }
   return *this;
}

HybridIndexSet::~HybridIndexSet()
{
   for ( Index_type isi = 0; isi < getNumParts(); ++isi ) {
      const PartPair& is_pair = getPartPair(isi);

      switch ( is_pair.first ) {

         case _Range_ : {
            if ( is_pair.second ) {
               RangeIndexSet* is =
                  const_cast<RangeIndexSet*>(
                     static_cast<const RangeIndexSet*>(is_pair.second)
                  );
               delete is;
            }
            break;
         }

         case _Unstructured_ : {
            if ( is_pair.second ) {
               UnstructuredIndexSet* is =
                  const_cast<UnstructuredIndexSet*>(
                     static_cast<const UnstructuredIndexSet*>(is_pair.second)
                  );
               delete is;
            }
            break;
         }

         default : {
            /* fail */ ;
         }

      } // iterate over parts of hybrid index set
   }
}


/*
*************************************************************************
*
* Private helper function to copy hybrid index set parts.
*
*************************************************************************
*/
void HybridIndexSet::copyParts(const HybridIndexSet& other)
{
   for ( Index_type isi = 0; isi < other.getNumParts(); ++isi ) {
      const PartPair& is_pair = other.getPartPair(isi);

      switch ( is_pair.first ) {

         case _Range_ : {
            if ( is_pair.second ) {
               addIndexSet(
                  *static_cast<const RangeIndexSet*>(is_pair.second));
            }
            break;
         }

         case _Unstructured_ : {
            if ( is_pair.second ) {
               addIndexSet(
                  *static_cast<const UnstructuredIndexSet*>(is_pair.second));
            }
            break;
         }

         default : {
            /* fail */ ;
         }

      } // iterate over parts of hybrid index set
   }
}


void HybridIndexSet::addIndexSet(const RangeIndexSet& index_set)
{
   RangeIndexSet* new_is = 
      new RangeIndexSet(index_set.getBegin(), index_set.getEnd());
   m_indexsets.push_back( std::make_pair( _Range_, new_is ) );

   m_len += new_is->getLength();
}

void HybridIndexSet::addRangeIndices(Index_type begin, Index_type end)
{
   RangeIndexSet* new_is = new RangeIndexSet(begin, end);
   m_indexsets.push_back( std::make_pair( _Range_, new_is ) );

   m_len += new_is->getLength();
}

void HybridIndexSet::addIndexSet(const UnstructuredIndexSet& index_set)
{
   UnstructuredIndexSet* new_is = 
      new UnstructuredIndexSet(index_set.getIndex(),index_set.getLength());
   m_indexsets.push_back( std::make_pair( _Unstructured_, new_is ) );

   m_len += new_is->getLength();
}

void HybridIndexSet::addUnstructuredIndices(const Index_type* indx, Index_type len)
{
   UnstructuredIndexSet* new_is = new UnstructuredIndexSet(indx, len);
   m_indexsets.push_back( std::make_pair( _Unstructured_, new_is ) );

   m_len += new_is->getLength();
}

#endif

}  // closing brace for RAJA namespace
