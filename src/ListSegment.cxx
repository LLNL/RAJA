/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for list segment classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/ListSegment.hxx"

#include <iostream>

#if !defined(RAJA_USE_STL)
#include <cstdio>
#include <cstring>
#endif

#if defined(RAJA_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace RAJA {


/*
*************************************************************************
*
* Public ListSegment class methods.
*
*************************************************************************
*/

////
////
ListSegment::ListSegment(const Index_type* indx, Index_type len,
                         IndexOwnership indx_own)
: BaseSegment( _ListSeg_ )
{
   initIndexData(indx, len, indx_own);
}

////
////
ListSegment::ListSegment(const ListSegment& other)
: BaseSegment( _ListSeg_ )
{
   initIndexData(other.m_indx, other.getLength(), other.m_indx_own);
}

////
////
ListSegment& ListSegment::operator=(const ListSegment& rhs)
{
   if ( &rhs != this ) {
      ListSegment copy(rhs);
      this->swap(copy);
   }
   return *this;
}

////
////
ListSegment::~ListSegment()
{
   if ( m_indx && m_indx_own == Owned ) {
#if defined(RAJA_USE_CUDA)
      if (cudaFree(m_indx) != cudaSuccess) {
         std::cerr << "\n ERROR in cudaFree call, FILE: "
                      << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
       }
#else
      delete[] m_indx ;
#endif
   }
}

////
////
void ListSegment::swap(ListSegment& other)
{
#if defined(RAJA_USE_STL)
   using std::swap;
   swap(m_indx, other.m_indx);
   swap(m_len, other.m_len);
   swap(m_indx_own, other.m_indx_own);
#else
   Index_type* tindx        = m_indx;
   Index_type  tlen         = m_len;
   IndexOwnership tindx_own = m_indx_own;

   m_indx     = other.m_indx;
   m_len      = other.m_len;
   m_indx_own = other.m_indx_own;

   other.m_indx     = tindx;
   other.m_len      = tlen;
   other.m_indx_own = tindx_own;
#endif
}

////
////
bool ListSegment::indicesEqual(const Index_type* indx, Index_type len) const
{
   bool equal = true;

   if ( len != m_len || indx == 0 || m_indx == 0 ) {

      equal = false;

   } else {

     Index_type i = 0;
     while ( equal && i < m_len ) {
        equal =  (m_indx[i] == indx[i]) ;
        i++;
     }

   }

   return equal;
}

////
////
void ListSegment::print(std::ostream& os) const
{
   os << "ListSegment : length, owns index = " << getLength() 
      << (m_indx_own == Owned ? " -- Owned" : " -- Unowned") << std::endl;
   for (Index_type i = 0; i < getLength(); ++i) {
      os << "\t" << m_indx[i] << std::endl;
   }
}

/*
*************************************************************************
*
* Private initialization method.
*
*************************************************************************
*/
void ListSegment::initIndexData(const Index_type* indx, 
                                Index_type len,
                                IndexOwnership indx_own)
{
   if ( len <= 0 || indx == 0 ) {

      m_indx = 0;
      m_len  = 0;
      m_indx_own = Unowned;

   } else { 

      m_len = len;
      m_indx_own = indx_own;

      if ( m_indx_own == Owned ) {
#if defined(RAJA_USE_CUDA)

         if ( cudaMallocManaged((void **)&m_indx, m_len*sizeof(Index_type), 
                                 cudaMemAttachGlobal) != cudaSuccess ) {
            std::cerr << "\n ERROR in cudaMallocManaged call, FILE: " 
                      << __FILE__ << " line " << __LINE__ << std::endl;
            exit(1);
         } 
         cudaMemset(m_indx,0,m_len*sizeof(Index_type));

#else
         m_indx = new Index_type[len];
#endif

         for (Index_type i = 0; i < m_len; ++i) {
            m_indx[i] = indx[i] ;
         }

      } else {
         // Uh-oh. Using evil const_cast.... 
         m_indx = const_cast<Index_type*>(indx);
      }

   } 
}


}  // closing brace for RAJA namespace
