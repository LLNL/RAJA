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
 * \brief   Implementation file for Box segment classes
 *
 ******************************************************************************
 */

#include "RAJA/BoxSegment.hxx"

#include <iostream>

namespace RAJA {


/*
*************************************************************************
*
* BoxSegment class methods
*
*************************************************************************
*/

void BoxSegment::print(std::ostream& os) const
{
   os << "BoxSegment : length = " << getLength() 
      << " : corner, dim = "
      << m_corner << ", " << m_dim << std::endl;
}


}  // closing brace for RAJA namespace
