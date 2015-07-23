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

#include "IndexSet.hxx"

#include <cstdlib>
#include <iostream>


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
#if USE_CUDA
    if (cudaMallocManaged((void **)&m_indx, m_len*sizeof(Index_type), cudaMemAttachGlobal) 
								!= cudaSuccess)
    {
      std::cerr << "\n ERROR in CUDA Call, FILE: " << __FILE__ << " line "
		<< __LINE__ << std::endl;
      exit(1);
    }
#else
      posix_memalign((void **)&m_indx,
                     IndexSet_VECTOR_ALIGN, m_len*sizeof(Index_type)) ;
		     
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
#if USE_CUDA
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
#if USE_CUDA
      if (cudaMallocManaged((void **)&tindx, len*sizeof(Index_type), cudaMemAttachGlobal)
						           != cudaSuccess) 
      {
	std::cerr << "\n ERROR line " << __LINE__ 
		  << " cudaMallocManaged failed!" << std::endl;
	exit(1);
      }	
#else
      posix_memalign((void **)&tindx,
                     IndexSet_VECTOR_ALIGN, len*sizeof(Index_type)) ;
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

#if 0
/*
*************************************************************************
*
* HybridIndexSet builder methods.
*
*************************************************************************
*/

HybridIndexSet* IndexSet_build(const std::vector<Index_type>& indices)
{
   return( IndexSet_build(&indices[0], indices.size()) );
}

HybridIndexSet* IndexSet_build(const Index_type* const indices_in, 
                               Index_type length)
{
   HybridIndexSet* hindex = new HybridIndexSet();

   /* only transform relatively large */
   if (length > IndexSet_STRUCT_RANGE_MIN_LENGTH) {
      /* build a rindex array from an index array */
      Index_type docount = 0 ;
      Index_type inrange = -1 ;

      /****************************/
      /* first, gather statistics */
      /****************************/

      Index_type scanVal = indices_in[0] ;
      Index_type sliceCount = 0 ;
      for (Index_type ii=1; ii<length; ++ii) {
         Index_type lookAhead = indices_in[ii] ;

         if (inrange == -1) {
            if ( (lookAhead == scanVal+1) && 
                 ((scanVal % IndexSet_VECTOR_DOUBLE_ALIGN) == 0) ) {
              inrange = 1 ;
            }
            else {
              inrange = 0 ;
            }
         }

         if (lookAhead == scanVal+1) {
            if ( (inrange == 0) && ((scanVal % IndexSet_VECTOR_DOUBLE_ALIGN) == 0) ) {
               if (sliceCount != 0) {
                  docount += 1 + sliceCount ; /* length + singletons */
               }
               inrange = 1 ;
               sliceCount = 0 ;
            }
            ++sliceCount ;  /* account for scanVal */
         }
         else {
            if (inrange == 1) {
               /* we can tighten this up by schleping any trailing */
               /* sigletons off into the subsequent singleton */
               /* array.  We would then also need to recheck the */
               /* final length of the range to make sure it meets */
               /* our minimum length crietria.  If it doesnt, */
               /* we need to emit a random array instead of */
               /* a range array */
               ++sliceCount ;
               docount += 2 ; /* length + begin */
               inrange = 0 ;
               sliceCount = 0 ;
            }
            else {
              ++sliceCount ;  /* account for scanVal */
            }
         }

         scanVal = lookAhead ;
      }  // end loop to gather statistics

      if (inrange != -1) {
         if (inrange) {
            ++sliceCount ;
            docount += 2 ; /* length + begin */
         }
         else {
            ++sliceCount ;
            docount += 1 + sliceCount ; /* length + singletons */
         }
      }
      else if (scanVal != -1) {
         ++sliceCount ;
         docount += 2 ;
      }
      ++docount ; /* zero length termination */

      /* What is the cutoff criteria for generating the rindex array? */
      if (docount < (length*(IndexSet_VECTOR_DOUBLE_ALIGN-1))/IndexSet_VECTOR_DOUBLE_ALIGN) {
         /* The rindex array can either contain a pointer into the */
         /* original index array, *or* it can repack the data from the */
         /* original index array.  Benefits of repacking could include */
         /* better use of hardware prefetch streams, or guaranteeing */
         /* alignment of index array segments. */

         /*******************************/
         /* now, build the rindex array */
         /*******************************/

         Index_type dobegin ;
         inrange = -1 ;

         scanVal = indices_in[0] ;
         sliceCount = 0 ;
         dobegin = scanVal ;
         for (Index_type ii=1; ii < length; ++ii) {
            Index_type lookAhead = indices_in[ii] ;

            if (inrange == -1) {
               if ( (lookAhead == scanVal+1) && 
                    ((scanVal % IndexSet_VECTOR_DOUBLE_ALIGN) == 0) ) {
                 inrange = 1 ;
               }
               else {
                 inrange = 0 ;
                 dobegin = ii-1 ;
               }
            }
            if (lookAhead == scanVal+1) {
               if ( (inrange == 0) && 
                    ((scanVal % IndexSet_VECTOR_DOUBLE_ALIGN) == 0) ) {
                  if (sliceCount != 0) {
                     hindex->addUnstructuredIndices(&indices_in[dobegin], 
                                                    sliceCount);
                  }
                  inrange = 1 ;
                  dobegin = scanVal ;
                  sliceCount = 0 ;
               }
               ++sliceCount ;  /* account for scanVal */
            }
            else {
               if (inrange == 1) {
               /* we can tighten this up by schleping any trailing */
               /* sigletons off into the subsequent singleton */
               /* array.  We would then also need to recheck the */
               /* final length of the range to make sure it meets */
               /* our minimum length crietria.  If it doesnt, */
               /* we need to emit a random array instead of */
               /* a range array */
                  ++sliceCount ;
                  hindex->addRangeIndices(dobegin, dobegin+sliceCount);
                  inrange = 0 ;
                  sliceCount = 0 ;
                  dobegin = ii ;
               }
               else {
                 ++sliceCount ;  /* account for scanVal */
               }
            }

            scanVal = lookAhead ;
         }  // for (Index_type ii ...

         if (inrange != -1) {
            if (inrange) {
               ++sliceCount ;
               hindex->addRangeIndices(dobegin, dobegin+sliceCount);
            }
            else {
               ++sliceCount ;
               hindex->addUnstructuredIndices(&indices_in[dobegin], sliceCount);
            }
         }
         else if (scanVal != -1) {
            hindex->addUnstructuredIndices(&scanVal, 1);
         }
      }
      else {  // !(docount < (length*IndexSet_VECTOR_DOUBLE_ALIGN-1))/IndexSet_VECTOR_DOUBLE_ALIGN)
         hindex->addUnstructuredIndices(indices_in, length);
      }
   }
   else {  // else !(length > IndexSet_STRUCT_RANGE_MIN_LENGTH)
      hindex->addUnstructuredIndices(indices_in, length);
   }

   return( hindex );
}
#endif
