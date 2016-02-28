/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for range segment classes
 *
 ******************************************************************************
 */

#include "RAJA/core/RangeSegment.hxx"

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
