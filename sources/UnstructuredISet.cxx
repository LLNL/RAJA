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

#include <cstdlib>
#include <iostream>
#include <algorithm>


namespace RAJA {


/*
*************************************************************************
*
* Public UnstructuredISet class methods.
*
*************************************************************************
*/

UnstructuredISet::UnstructuredISet(const Index_type* indx, Index_type len,
                                   bool owns_index)
{
   initIndexData(indx, len, owns_index);
}

UnstructuredISet::UnstructuredISet(const UnstructuredISet& other)
{
   initIndexData(other.m_indx, other.m_len, other.m_owns_index);
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
   if (m_owns_index && m_indx) {
      delete[] m_indx ;
   }
}

void UnstructuredISet::swap(UnstructuredISet& other)
{
   using std::swap;
   swap(m_len, other.m_len);
   swap(m_indx, other.m_indx);
   swap(m_owns_index, other.m_owns_index);
}

void UnstructuredISet::print(std::ostream& os) const
{
   os << "\nUnstructuredISet : length, owns index = "
      << m_len << " , " << m_owns_index << std::endl;
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
                                     bool owns_index)
{
   if ( len <= 0 ) {

      m_indx = 0;
      m_len = 0;
      m_owns_index = false;

   } else { 

      m_len = len;
      m_owns_index = owns_index;

      if ( m_owns_index ) {
         m_indx = new Index_type[len];
         std::copy(indx, indx + m_len, m_indx);
      } else {
         // Uh-oh. Using evil const_cast.... 
         m_indx = const_cast<Index_type*>(indx);
      }

   } 
}


}  // closing brace for namespace statement
