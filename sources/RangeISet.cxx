/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for range index set classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/RangeISet.hxx"

#include <iostream>

namespace RAJA {


/*
*************************************************************************
*
* RangeISet class methods
*
*************************************************************************
*/

void RangeISet::print(std::ostream& os) const
{
   os << "\nRangeISet::print : begin, end = "
      << m_begin << ", " << m_end << std::endl;
}


/*
*************************************************************************
*
* RangeStrideISet class methods
*
*************************************************************************
*/

void RangeStrideISet::print(std::ostream& os) const
{
   os << "\nRangeStrideISet::print : begin, end, stride = "
      << m_begin << ", " << m_end << ", " << m_stride << std::endl;
}


}  // closing brace for RAJA namespace
