/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for range segment classes
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/RangeSegment.hxx"

#include <iostream>

namespace RAJA {


/*
*************************************************************************
*
* RangeSegment class methods
*
*************************************************************************
*/

void RangeSegment::print(std::ostream& os) const
{
   os << "RangeSegment : length = " << getLength() 
      << " : begin, end = "
      << m_begin << ", " << m_end << std::endl;
}


/*
*************************************************************************
*
* RangeStrideSegment class methods
*
*************************************************************************
*/

void RangeStrideSegment::print(std::ostream& os) const
{
   os << "RangeStrideSegment : length = " << getLength() 
      << " : begin, end, stride = "
      << m_begin << ", " << m_end << ", " << m_stride << std::endl;
}


}  // closing brace for RAJA namespace
