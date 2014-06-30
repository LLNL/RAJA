/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for unstructured index set classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/UnstructuredISet.hxx"

#include <iostream>

#if !defined(RAJA_USE_STL)
#include <cstdio>
#include <cstring>
#endif

namespace RAJA {


/*
*************************************************************************
*
* Public UnstructuredISet class methods.
*
*************************************************************************
*/

UnstructuredISet::UnstructuredISet(const Index_type* indx, Index_type len,
                                   IndexOwnership indx_own)
{
   initIndexData(indx, len, indx_own);
}

UnstructuredISet::UnstructuredISet(const UnstructuredISet& other)
{
   initIndexData(other.m_indx, other.m_len, other.m_indx_own);
}

UnstructuredISet& UnstructuredISet::operator=(const UnstructuredISet& rhs)
{
   if ( &rhs != this ) {
      UnstructuredISet copy(rhs);
      this->swap(copy);
   }
   return *this;
}

UnstructuredISet::~UnstructuredISet()
{
   if ( m_indx && m_indx_own == Owned ) {
      delete[] m_indx ;
   }
}

void UnstructuredISet::swap(UnstructuredISet& other)
{
#if defined(RAJA_USE_STL)
   using std::swap;
   swap(m_indx, other.m_indx);
   swap(m_len, other.m_len);
   swap(m_indx_own, other.m_indx_own);
#else
   Index_type* tindx = m_indx;
   Index_type  tlen = m_len;
   IndexOwnership tindx_own = m_indx_own;

   m_indx = other.m_indx;
   m_len = other.m_len;
   m_indx_own = other.m_indx_own;

   other.m_indx = tindx;
   other.m_len = tlen;
   other.m_indx_own = tindx_own;
#endif
}

void UnstructuredISet::print(std::ostream& os) const
{
   os << "\nUnstructuredISet : length = " << m_len << " , owns index = "
      << (m_indx_own == Owned ? "Owned" : "Unowned") << std::endl;
   for (Index_type i = 0; i < m_len; ++i) {
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
void UnstructuredISet::initIndexData(const Index_type* indx, 
                                     Index_type len,
                                     IndexOwnership indx_own)
{
   if ( len <= 0 ) {

      m_indx = 0;
      m_len = 0;
      m_indx_own = Unowned;

   } else { 

      m_len = len;
      m_indx_own = indx_own;

      if ( m_indx_own == Owned ) {
         m_indx = new Index_type[len];
#if defined(RAJA_USE_STL)
         std::copy(indx, indx + m_len, m_indx);
#else
         memcpy(m_indx, indx, m_len*sizeof(Index_type));
#endif
      } else {
         // Uh-oh. Using evil const_cast.... 
         m_indx = const_cast<Index_type*>(indx);
      }

   } 
}


}  // closing brace for RAJA namespace
